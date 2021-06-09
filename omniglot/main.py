import argparse
import random
import numpy as np
import torch
import os
from data_generator import Omniglot
from maml import MAML

parser = argparse.ArgumentParser(description='MetaMix')
parser.add_argument('--datasource', default='omniglot', type=str, help='omniglot')
parser.add_argument('--num_classes', default=5, type=int,
                    help='number of classes used in classification (e.g. 5-way classification).')
parser.add_argument('--num_test_task', default=600, type=int, help='number of test tasks.')
parser.add_argument('--test_epoch', default=-1, type=int, help='test epoch, only work when test start')

## Training options
parser.add_argument('--metatrain_iterations', default=15000, type=int,
                    help='number of metatraining iterations.')  # 15k for omniglot, 50k for sinusoid
parser.add_argument('--meta_batch_size', default=25, type=int, help='number of tasks sampled per meta-update')
parser.add_argument('--update_lr', default=0.01, type=float, help='inner learning rate')
parser.add_argument('--meta_lr', default=0.001, type=float, help='the base learning rate of the generator')
parser.add_argument('--num_updates', default=5, type=int, help='num_updates in maml')
parser.add_argument('--num_updates_test', default=10, type=int, help='num_updates in maml')
parser.add_argument('--update_batch_size', default=5, type=int,
                    help='number of examples used for inner gradient update (K for K-shot learning).')
parser.add_argument('--update_batch_size_eval', default=15, type=int,
                    help='number of examples used for inner gradient test (K for K-shot learning).')
parser.add_argument('--num_filters', default=64, type=int,
                    help='number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')

## Logging, saving, and testing options
parser.add_argument('--logdir', default='xxx', type=str,
                    help='directory for summaries and checkpoints.')
parser.add_argument('--datadir', default='xxx', type=str, help='directory for datasets.')
parser.add_argument('--resume', default=0, type=int, help='resume training if there is a model available')
parser.add_argument('--train', default=1, type=int, help='True to train, False to test.')
parser.add_argument('--test_set', default=1, type=int,
                    help='Set to true to test on the the test set, False for the validation set.')
parser.add_argument('--shuffle', default=False, action='store_true', help='use channel shuffle or not')
parser.add_argument('--mix', default=False, action='store_true', help='use mixup or not')
parser.add_argument('--trial', default=0, type=int, help='trial')

args = parser.parse_args()
print(args)

assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True

random.seed(1)
np.random.seed(2)

exp_string = f'MetaMix.data_{args.datasource}.cls_{args.num_classes}.mbs_{args.meta_batch_size}.ubs_{args.update_batch_size}.metalr_{args.meta_lr}.innerlr_{args.update_lr}'

if args.num_filters != 64:
    exp_string += '.hidden' + str(args.num_filters)
if args.mix:
    exp_string += '.mix'
if args.shuffle:
    exp_string += '.shuffle'
if args.trial > 0:
    exp_string += '.trial_{}'.format(args.trial)

print(exp_string)

def train(args, maml, optimiser):
    Print_Iter = 100
    Save_Iter = 500
    print_loss, print_acc, print_loss_support = 0.0, 0.0, 0.0

    dataloader = Omniglot(args, 'train')

    if not os.path.exists(args.logdir + '/' + exp_string + '/'):
        os.makedirs(args.logdir + '/' + exp_string + '/')

    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(dataloader):
        if step > args.metatrain_iterations:
            break
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).cuda(), y_spt.squeeze(0).cuda(), \
                                     x_qry.squeeze(0).cuda(), y_qry.squeeze(0).cuda()
        task_losses = []
        task_acc = []

        if step<2000:
            is_shuffle = 0
        else:
            is_shuffle = np.random.randint(2)

        for meta_batch in range(args.meta_batch_size):
            if args.shuffle and is_shuffle == 1:
                loss_val, acc_val = maml.forward_channel_shuffle(x_spt[meta_batch], y_spt[meta_batch],
                                                                 x_qry[meta_batch], y_qry[meta_batch])
            else:
                loss_val, acc_val = maml(x_spt[meta_batch], y_spt[meta_batch], x_qry[meta_batch], y_qry[meta_batch])
            task_losses.append(loss_val)
            task_acc.append(acc_val)

        meta_batch_loss = torch.stack(task_losses).mean()
        meta_batch_acc = torch.stack(task_acc).mean()

        optimiser.zero_grad()
        meta_batch_loss.backward()
        optimiser.step()

        if step != 0 and step % Print_Iter == 0:
            print('iter: {}, loss_all: {}, acc: {}'.format(
                    step, print_loss, print_acc))

            print_loss, print_acc = 0.0, 0.0
        else:
            print_loss += meta_batch_loss / Print_Iter
            print_acc += meta_batch_acc / Print_Iter

        if step != 0 and step % Save_Iter == 0:
            torch.save(maml.state_dict(),
                       '{0}/{2}/model{1}'.format(args.logdir, step, exp_string))


def test(args, maml, test_epoch):
    res_acc = []
    args.meta_batch_size=1

    dataloader = Omniglot(args, 'test')

    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(dataloader):
        if step > args.num_test_task:
            break
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).cuda(), y_spt.squeeze(0).cuda(), \
                                     x_qry.squeeze(0).cuda(), y_qry.squeeze(0).cuda()
        _, acc_val = maml(x_spt, y_spt, x_qry, y_qry)
        res_acc.append(acc_val.item())

    res_acc = np.array(res_acc)

    print('test_epoch is {}, acc is {}, ci95 is {}'.format(test_epoch, np.mean(res_acc),
                                         1.96 * np.std(res_acc) / np.sqrt(args.num_test_task * args.meta_batch_size)))


def main():
    maml = MAML(args).cuda()

    if args.resume == 1 and args.train == 1:
        model_file = '{0}/{2}/model{1}'.format(args.logdir, args.test_epoch, exp_string)
        print(model_file)
        maml.load_state_dict(torch.load(model_file))

    meta_optimiser = torch.optim.Adam(list(maml.parameters()),
                                      lr=args.meta_lr, weight_decay=args.weight_decay)

    if args.train == 1:
        train(args, maml, meta_optimiser)
    else:
        for test_epoch in range(5000, 50000, 500):
            try:
                model_file = '{0}/{2}/model{1}'.format(args.logdir, test_epoch, exp_string)
                maml.load_state_dict(torch.load(model_file))
                test(args, maml, test_epoch)
            except IOError:
                continue


if __name__ == '__main__':
    main()