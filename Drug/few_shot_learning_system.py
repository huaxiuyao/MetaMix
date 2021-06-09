import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from meta_neural_network_architectures import FCNReLUNormNetwork
from inner_loop_optimizers import LSLRGradientDescentLearningRule

from utils.experiment import pearson_score

def set_torch_seed(seed):
    """
    Sets the pytorch seeds for current experiment run
    :param seed: The seed (int)
    :return: A random number generator to use
    """
    rng = np.random.RandomState(seed=seed)
    torch_seed = rng.randint(0, 999999)
    torch.manual_seed(seed=torch_seed)

    return rng


class MAMLRegressor(nn.Module):
    def __init__(self, input_shape, args):

        super(MAMLRegressor, self).__init__()
        self.args = args
        self.batch_size = args.meta_batch_size
        self.current_epoch = 0
        self.input_shape = input_shape

        self.mixup = self.args.mixup

        self.rng = set_torch_seed(seed=0)

        self.args.rng = self.rng
        self.regressor = FCNReLUNormNetwork(input_shape=self.input_shape, args=self.args, meta=True).cuda()

        self.task_learning_rate = args.update_lr

        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(init_learning_rate=self.task_learning_rate,
                                                                    total_num_inner_loop_steps=self.args.num_updates,
                                                                    use_learnable_learning_rates=self.args.learnable_per_layer_per_step_inner_loop_learning_rate)
        self.inner_loop_optimizer.initialise(
            names_weights_dict=self.get_inner_loop_parameter_dict(params=self.regressor.named_parameters()))

        print("Inner Loop parameters")
        for key, value in self.inner_loop_optimizer.named_parameters():
            print(key, value.shape)

        self.args = args
        self.cuda()
        print("Outer Loop parameters")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.shape, param.requires_grad)


        self.optimizer = optim.Adam(self.trainable_parameters(), lr=args.meta_lr, amsgrad=False)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.args.metatrain_iterations,
                                                              eta_min=self.args.min_learning_rate)

    def get_per_step_loss_importance_vector(self):

        loss_weights = np.ones(shape=(self.args.num_updates)) * (
                1.0 / self.args.num_updates)
        decay_rate = 1.0 / self.args.num_updates / self.args.multi_step_loss_num_epochs
        min_value_for_non_final_losses = 0.03 / self.args.num_updates
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (self.current_epoch * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] + (self.current_epoch * (self.args.num_updates - 1) * decay_rate),
            1.0 - ((self.args.num_updates - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).cuda()
        return loss_weights

    def get_inner_loop_parameter_dict(self, params):

        param_dict = dict()
        for name, param in params:
            if param.requires_grad:
                if self.args.enable_inner_loop_optimizable_bn_params:
                    param_dict[name] = param.cuda()
                else:
                    #if "norm_layer" not in name:
                    #    param_dict[name] = param.cuda()

                    if "layer_dict.linear.bias" in name or "layer_dict.linear.weights" in name:
                        param_dict[name] = param.cuda()

        return param_dict

    def apply_inner_loop_update(self, loss, names_weights_copy, use_second_order, current_step_idx):

        self.regressor.zero_grad(names_weights_copy)

        grads = torch.autograd.grad(loss, names_weights_copy.values(),
                                    create_graph=use_second_order)
        names_grads_wrt_params = dict(zip(names_weights_copy.keys(), grads))

        names_weights_copy = self.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy,
                                                                     names_grads_wrt_params_dict=names_grads_wrt_params,
                                                                     num_step=current_step_idx)

        return names_weights_copy

    def get_across_task_loss_metrics(self, total_losses, total_accuracies):
        losses = dict()

        losses['loss'] = torch.mean(torch.stack(total_losses))
        if sum(~np.isnan(total_accuracies)) == 0:
            losses['accuracy'] = 0
        else:
            losses['accuracy'] = np.mean(np.array(total_accuracies)[~np.isnan(total_accuracies)])

        return losses

    def forward(self, data_batch, epoch, use_second_order, use_multi_step_loss_optimization, num_steps, training_phase):

        support_set_x, support_set_y, support_set_z, support_set_assay, \
        target_set_x, target_set_y, target_set_z, target_set_assay, seed = data_batch

        b = len(support_set_y)


        total_losses = []
        total_accuracies = []
        per_task_target_preds = [[] for i in range(len(target_set_x))]
        self.regressor.zero_grad()

        for task_id, (support_set_x_task, support_set_y_task, support_set_z_task, support_set_assay_task,
                      target_set_x_task, target_set_y_task, target_set_z_task, target_set_assay_task,) in \
                enumerate(zip(support_set_x,
                              support_set_y,
                              support_set_z,
                              support_set_assay,
                              target_set_x,
                              target_set_y,
                              target_set_z,
                              target_set_assay)):

            # first of all, put all tensors to the device
            support_set_x_task = torch.Tensor(support_set_x_task[0]).float().cuda()
            support_set_y_task = torch.Tensor(support_set_y_task[0]).float().cuda()
            support_set_z_task = torch.IntTensor(support_set_z_task[0]).int().cuda()
            support_set_assay_task = torch.LongTensor(support_set_assay_task).int().cuda()
            target_set_x_task = torch.Tensor(target_set_x_task[0]).float().cuda()
            target_set_y_task = torch.Tensor(target_set_y_task[0]).float().cuda()
            target_set_z_task = torch.IntTensor(target_set_z_task[0]).int().cuda()
            target_set_assay_task = torch.LongTensor(target_set_assay_task).int().cuda()

            if support_set_x_task.shape[0]>2000:
                sel_id = np.random.choice(np.arange(support_set_x_task.shape[0]), 2000, replace=False)
                support_set_x_task = support_set_x_task[sel_id]
                support_set_y_task = support_set_y_task[sel_id]

            if target_set_x_task.shape[0]>2000:
                sel_id = np.random.choice(np.arange(target_set_x_task.shape[0]), 2000, replace=False)
                target_set_x_task = target_set_x_task[sel_id]
                target_set_y_task = target_set_y_task[sel_id]

            task_losses = []
            task_accuracies = []
            per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector()
            names_weights_copy = self.get_inner_loop_parameter_dict(self.regressor.named_parameters())

            ns, fp_dim = support_set_x_task.shape
            nt, _ = target_set_x_task.shape

            for num_step in range(num_steps):

                support_loss, support_preds = self.net_forward(x=support_set_x_task,
                                                               y=support_set_y_task,
                                                               weights=names_weights_copy,
                                                               backup_running_statistics=
                                                               True if (num_step == 0) else False,
                                                               training=True, num_step=num_step)

                names_weights_copy = self.apply_inner_loop_update(loss=support_loss,
                                                                  names_weights_copy=names_weights_copy,
                                                                  use_second_order=use_second_order,
                                                                  current_step_idx=num_step)

                if use_multi_step_loss_optimization and training_phase and epoch < self.args.multi_step_loss_num_epochs:

                    # mixup here
                    if self.mixup:
                        indices = torch.randperm(ns).cuda()

                        lam = self.rng.beta(self.args.alpha, self.args.alpha)
                        mixed_set_x_task = torch.cat((support_set_x_task[indices,:][:nt,:], target_set_x_task), dim=0)
                        mixed_set_y_task = lam * support_set_y_task[indices][:nt] + (1-lam) * target_set_y_task



                        target_loss, target_preds = self.net_forward(x=mixed_set_x_task,
                                                                     y=mixed_set_y_task, weights=names_weights_copy,
                                                                     backup_running_statistics=False, training=True,
                                                                     num_step=num_step, mixup=self.mixup, lam=lam)
                    else:
                        target_loss, target_preds = self.net_forward(x=target_set_x_task,
                                                                     y=target_set_y_task, weights=names_weights_copy,
                                                                     backup_running_statistics=False, training=True,
                                                                     num_step=num_step, mixup=self.mixup)

                    task_losses.append(per_step_loss_importance_vectors[num_step] * target_loss)
                else:
                    if num_step == (self.args.num_updates - 1) and training_phase:
                        if self.mixup:
                            indices = torch.randperm(ns).cuda()
                            lam = self.rng.beta(self.args.alpha, self.args.alpha)
                            mixed_set_x_task = torch.cat((support_set_x_task[indices,:][:nt,:], target_set_x_task), dim=0)
                            mixed_set_y_task = lam * support_set_y_task[indices][:nt] + (1-lam) * target_set_y_task


                            target_loss, target_preds = self.net_forward(x=mixed_set_x_task,
                                                                         y=mixed_set_y_task, weights=names_weights_copy,
                                                                         backup_running_statistics=False, training=True,
                                                                         num_step=num_step, mixup=self.mixup, lam=lam)
                        else:
                            target_loss, target_preds = self.net_forward(x=target_set_x_task,
                                                                         y=target_set_y_task,
                                                                         weights=names_weights_copy,
                                                                         backup_running_statistics=False, training=True,
                                                                         num_step=num_step, mixup=self.mixup)
                        task_losses.append(target_loss)

            support_loss, support_preds = self.net_forward(x=support_set_x_task,
                                                                     y=support_set_y_task, weights=names_weights_copy,
                                                                     backup_running_statistics=False, training=True,
                                                                     num_step=num_steps-1)


            target_loss, target_preds = self.net_forward(x=target_set_x_task,
                                                                     y=target_set_y_task, weights=names_weights_copy,
                                                                     backup_running_statistics=False, training=True,
                                                                     num_step=num_steps-1)
            if not training_phase:        
                task_losses.append(target_loss)


            per_task_target_preds[task_id] = target_preds.detach().cpu().numpy()
            # _, predicted = torch.max(target_preds.data, 1)

            accuracy = pearson_score(target_set_y_task.detach().cpu().numpy(),target_preds.detach().cpu().numpy())
            task_losses = torch.sum(torch.stack(task_losses))
            total_losses.append(task_losses)
            total_accuracies.append(accuracy)

            if not training_phase:
                self.regressor.restore_backup_stats()

        losses = self.get_across_task_loss_metrics(total_losses=total_losses,
                                                   total_accuracies=total_accuracies)

        for idx, item in enumerate(per_step_loss_importance_vectors):
            losses['loss_importance_vector_{}'.format(idx)] = item.detach().cpu().numpy()

        return losses, per_task_target_preds

    def net_forward(self, x, y, weights, backup_running_statistics, training, num_step, mixup=None, lam=None):

        preds = self.regressor.forward(x=x, params=weights,
                                        training=training,
                                        backup_running_statistics=backup_running_statistics, num_step=num_step, mixup=mixup, lam=lam)

        if mixup:
            npreds = preds.shape[0]
            preds = preds[:int(npreds/2),:]

        loss = F.mse_loss(input=preds, target=y.unsqueeze(dim=-1))


        return loss, preds

    def trainable_parameters(self):

        for param in self.parameters():
            if param.requires_grad:
                yield param

    def train_forward_prop(self, data_batch, epoch):

        losses, per_task_target_preds = self.forward(data_batch=data_batch, epoch=epoch,
                                                     use_second_order=self.args.second_order and
                                                                      epoch > self.args.first_order_to_second_order_epoch,
                                                     use_multi_step_loss_optimization=self.args.use_multi_step_loss_optimization,
                                                     num_steps=self.args.num_updates,
                                                     training_phase=True)
        return losses, per_task_target_preds

    def evaluation_forward_prop(self, data_batch, epoch):

        losses, per_task_target_preds = self.forward(data_batch=data_batch, epoch=epoch, use_second_order=False,
                                                     use_multi_step_loss_optimization=True,
                                                     num_steps=self.args.test_num_updates,
                                                     training_phase=False)

        return losses, per_task_target_preds

    def meta_update(self, loss):

        self.optimizer.zero_grad()
        loss.backward()
        if 'imagenet' in self.args.dataset_name:
            for name, param in self.regressor.named_parameters():
                if param.requires_grad:
                    param.grad.data.clamp_(-10, 10)  # not sure if this is necessary, more experiments are needed
        self.optimizer.step()

    def run_train_iter(self, data_batch, epoch):

        epoch = int(epoch)
        self.scheduler.step(epoch=epoch)
        if self.current_epoch != epoch:
            self.current_epoch = epoch

        if not self.training:
            self.train()

        losses, per_task_target_preds = self.train_forward_prop(data_batch=data_batch, epoch=epoch)

        self.meta_update(loss=losses['loss'])
        losses['learning_rate'] = self.scheduler.get_lr()[0]
        self.optimizer.zero_grad()
        self.zero_grad()

        return losses, per_task_target_preds

    def run_validation_iter(self, data_batch):

        if self.training:
            self.eval()


        losses, per_task_target_preds = self.evaluation_forward_prop(data_batch=data_batch, epoch=self.current_epoch)


        return losses, per_task_target_preds

    def save_model(self, model_save_dir, state):

        state['network'] = self.state_dict()
        torch.save(state, f=model_save_dir)

    def load_model(self, model_save_dir, model_name, model_idx):

        filepath = os.path.join(model_save_dir, "{}_{}".format(model_name, model_idx))
        state = torch.load(filepath)
        state_dict_loaded = state['network']
        self.load_state_dict(state_dict=state_dict_loaded)
        return state
