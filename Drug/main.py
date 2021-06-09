import argparse
import random

import ipdb
import numpy as np
import torch
from torch.utils.data import DataLoader
import os

from data import OurMetaDataset, MetaLearningSystemDataLoader
from few_shot_learning_system import MAMLRegressor
from scipy.stats import pearsonr

parser = argparse.ArgumentParser(description='MetaMix')
parser.add_argument('--datasource', default='drug', type=str, help='drug')
parser.add_argument('--dim_w', default=1024, type=int, help='dimension of w')
parser.add_argument('--hid_dim', default=500, type=int, help='dimension of w')
parser.add_argument('--num_stages', default=2, type=int, help='num stages')
parser.add_argument('--per_step_bn_statistics', default=True, action='store_false')
parser.add_argument('--learnable_bn_gamma', default=True, action='store_false', help='learnable_bn_gamma')
parser.add_argument('--learnable_bn_beta', default=True, action='store_false', help='learnable_bn_beta')
parser.add_argument('--enable_inner_loop_optimizable_bn_params', default=False, action='store_true', help='enable_inner_loop_optimizable_bn_params')
parser.add_argument('--learnable_per_layer_per_step_inner_loop_learning_rate', default=True, action='store_false', help='learnable_per_layer_per_step_inner_loop_learning_rate')
parser.add_argument('--use_multi_step_loss_optimization', default=True, action='store_false', help='use_multi_step_loss_optimization')
parser.add_argument('--second_order', default=1, type=int, help='second_order')
parser.add_argument('--first_order_to_second_order_epoch', default=-1, type=int, help='first_order_to_second_order_epoch')
parser.add_argument('--mixup', default=False, action='store_true', help='metamix')

parser.add_argument('--dim_y', default=1, type=int, help='dimension of w')
parser.add_argument('--dataset_name', default='assay', type=str,
                    help='dataset_name.')
parser.add_argument('--dataset_path', default='ci9b00375_si_002.txt', type=str,
                    help='dataset_path.')
parser.add_argument('--type_filename', default='ci9b00375_si_001.txt', type=str,
                    help='type_filename.')
parser.add_argument('--compound_filename', default='ci9b00375_si_003.txt', type=str,
                    help='Directory of data files.')

parser.add_argument('--fp_filename', default='compound_fp.npy', type=str,
                    help='fp_filename.')

parser.add_argument('--target_assay_list', default='591252', type=str,
                    help='target_assay_list')

parser.add_argument('--train_seed', default=0, type=int, help='train_seed')
parser.add_argument('--val_seed', default=0, type=int, help='val_seed')
parser.add_argument('--test_seed', default=0, type=int, help='test_seed')

parser.add_argument('--train_val_split', default=[0.9588, 0.0177736202, 0.023386342], type=list, help='train_val_split')
parser.add_argument('--num_evaluation_tasks', default=100, type=int, help='num_evaluation_tasks')
parser.add_argument('--drug_group', default=-1, type=int, help='drug group')
parser.add_argument('--metatrain_iterations', default=20, type=int,
                    help='number of metatraining iterations.')  # 15k for omniglot, 50k for sinusoid
parser.add_argument('--meta_batch_size', default=4, type=int, help='number of tasks sampled per meta-update')
parser.add_argument('--min_learning_rate', default=0.00001, type=float, help='min_learning_rate')
parser.add_argument('--update_lr', default=0.01, type=float, help='inner learning rate')
parser.add_argument('--meta_lr', default=0.001, type=float, help='the base learning rate of the generator')
parser.add_argument('--num_updates', default=5, type=int, help='num_updates in maml')
parser.add_argument('--test_num_updates', default=5, type=int, help='num_updates in maml')
parser.add_argument('--multi_step_loss_num_epochs', default=5, type=int, help='multi_step_loss_num_epochs')
parser.add_argument('--norm_layer', default='batch_norm', type=str, help='norm_layer')
parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
parser.add_argument('--alpha', default=0.5, type=float, help='alpha in beta distribution')

## Logging, saving, and testing options
parser.add_argument('--logdir', default='xxx', type=str,
                    help='directory for summaries and checkpoints.')
parser.add_argument('--datadir', default='xxx', type=str, help='directory for datasets.')
parser.add_argument('--resume', default=0, type=int, help='resume training if there is a model available')
parser.add_argument('--train', default=1, type=int, help='True to train, False to test.')
parser.add_argument('--test_epoch', default=-1, type=int, help='test epoch, only work when test start')
parser.add_argument('--trial', default=0, type=int, help='trial for each layer')

args = parser.parse_args()
print(args)

assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True

random.seed(1)
np.random.seed(2)

exp_string = f'MetaMix.data_{args.datasource}.mbs_{args.meta_batch_size}.metalr_{args.meta_lr}.innerlr_{args.update_lr}.drug_group_{args.drug_group}'

if args.trial > 0:
    exp_string += '.trial{}'.format(args.trial)
if args.mixup:
    exp_string += '.mix'

print(exp_string)

args.datadir = args.datadir+ '/' +args.datasource

def train(args, maml, optimiser, dataloader):
    Print_Iter = 100
    Save_Iter = 500
    print_loss, print_acc, print_loss_support = 0.0, 0.0, 0.0

    data_each_epoch = 4100

    if not os.path.exists(args.logdir + '/' + exp_string + '/'):
        os.makedirs(args.logdir + '/' + exp_string + '/')

    for epoch in range(args.metatrain_iterations):
        train_data_all = dataloader.get_train_batches()
        for step, cur_data in enumerate(train_data_all):

            meta_batch_loss, _ = maml.run_train_iter(cur_data, epoch)

            if step != 0 and step % Print_Iter == 0:
                print('epoch: {}, iter: {}, mse: {}'.format(epoch, step, print_loss))

                print_loss = 0.0
            else:
                print_loss += meta_batch_loss['loss'] / Print_Iter

            if step != 0 and step % Save_Iter == 0:
                torch.save(maml.state_dict(),
                           '{0}/{2}/model{1}'.format(args.logdir, step+epoch*data_each_epoch, exp_string))


def test(args, epoch, maml, dataloader):
    res_acc = []

    valid_cnt = 0


    test_data_all = dataloader.get_test_batches()

    for step, cur_data in enumerate(test_data_all):

        loss, _  = maml.run_validation_iter(cur_data)
        r2 = loss['accuracy']

        res_acc.append(r2)

        if r2 > 0.3:
            valid_cnt += 1

    res_acc = np.array(res_acc)
    median = np.median(res_acc, 0)
    mean = np.mean(res_acc, 0)

    print('epoch is: {} mean is: {}, median is: {}, cnt>0.3 is: {}'.format(epoch, mean, median, valid_cnt))


def main():
    maml = MAMLRegressor(args=args, input_shape=(2, args.dim_w))

    if args.resume == 1 and args.train == 1:
        model_file = '{0}/{2}/model{1}'.format(args.logdir, args.test_epoch, exp_string)
        print(model_file)
        maml.load_state_dict(torch.load(model_file))

    meta_optimiser = torch.optim.Adam(list(maml.parameters()),
                                      lr=args.meta_lr, weight_decay=args.weight_decay)

    dataloader = MetaLearningSystemDataLoader(args, target_assay=args.target_assay_list)

    if args.train == 1:
        train(args, maml, meta_optimiser, dataloader)
    else:
        args.meta_batch_size = 1
        model_file = '{0}/{2}/model{1}'.format(args.logdir, args.test_epoch, exp_string)
        maml.load_state_dict(torch.load(model_file))
        test(args, args.test_epoch, maml, dataloader)



if __name__ == '__main__':
    main()
