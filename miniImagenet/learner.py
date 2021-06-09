import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.distributions import Beta

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Conv_Standard(nn.Module):
    def __init__(self, args, x_dim, hid_dim, z_dim, final_layer_size):
        super(Conv_Standard, self).__init__()
        self.args = args
        self.net = nn.Sequential(self.conv_block(x_dim, hid_dim), self.conv_block(hid_dim, hid_dim),
                                 self.conv_block(hid_dim, hid_dim), self.conv_block(hid_dim, z_dim), Flatten())
        self.dist = Beta(torch.FloatTensor([2]), torch.FloatTensor([2]))
        self.hid_dim = hid_dim

        self.logits = nn.Linear(final_layer_size, self.args.num_classes)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def functional_conv_block(self, x, weights, biases,
                              bn_weights, bn_biases, is_training):

        x = F.conv2d(x, weights, biases, padding=1)
        x = F.batch_norm(x, running_mean=None, running_var=None, weight=bn_weights, bias=bn_biases,
                         training=is_training)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x

    def mixup_data(self, xs, ys, xq, yq):
        query_size = xq.shape[0]

        shuffled_index = torch.randperm(query_size)

        xs = xs[shuffled_index]
        ys = ys[shuffled_index]
        lam = self.dist.sample().cuda()
        mixed_x = lam * xq + (1 - lam) * xs

        return mixed_x, yq, ys, lam

    def forward(self, x):
        x = self.net(x)

        x = x.view(x.size(0), -1)

        return self.logits(x)

    def functional_forward(self, x, weights, is_training=True):
        for block in range(4):
            x = self.functional_conv_block(x, weights[f'net.{block}.0.weight'], weights[f'net.{block}.0.bias'],
                                           weights.get(f'net.{block}.1.weight'), weights.get(f'net.{block}.1.bias'),
                                           is_training)

        x = x.view(x.size(0), -1)

        x = F.linear(x, weights['logits.weight'], weights['logits.bias'])

        return x

    def channel_shuffle(self, hidden, label, shuffle_dict, shuffle_channel_id):
        concept_idx = [0, 6, 11, 16, 22, 27, 32]

        new_data = []

        start = concept_idx[shuffle_channel_id]
        end = concept_idx[shuffle_channel_id + 1]

        for i in range(self.args.num_classes):
            cur_class_1 = hidden[label == i]
            cur_class_2 = hidden[label == shuffle_dict[i]]

            new_data.append(
                torch.cat((cur_class_1[:, :start], cur_class_2[:, start:end], cur_class_1[:, end:]), dim=1))

        new_data = torch.cat(new_data, dim=0)

        indexes = torch.randperm(new_data.shape[0])

        new_data = new_data[indexes]
        new_label = label[indexes]

        return new_data, new_label

    def forward_metamix(self, hidden_support, label_support, hidden_query, label_query, weights, is_training=True):

        sel_layer = random.randint(0, 3)
        flag = 0

        for layer in range(4):
            if layer==sel_layer:
                hidden_query, reweighted_query, reweighted_support, lam = self.mixup_data(hidden_support, label_support,
                                                                                       hidden_query, label_query)
                flag=1

            if not flag:
                hidden_support = self.functional_conv_block(hidden_support, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                                weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                                is_training)
            hidden_query = self.functional_conv_block(hidden_query, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                            weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                            is_training)

        hidden4_query = hidden_query.view(hidden_query.size(0), -1)

        x = F.linear(hidden4_query, weights['logits.weight'], weights['logits.bias'])

        return x, reweighted_query, reweighted_support, lam

    def functional_forward_cf(self, hidden, label, sel_layer, shuffle_list, shuffle_channel_id, weights,
                                           is_training=True):

        label_new = label

        for layer in range(4):
            if layer == sel_layer:
                hidden, label_new = self.channel_shuffle(hidden, label, shuffle_list, shuffle_channel_id)

            hidden = self.functional_conv_block(hidden, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                                weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                                is_training)

        hidden = hidden.view(hidden.size(0), -1)

        x = F.linear(hidden, weights['logits.weight'], weights['logits.bias'])

        return x, label_new

    def mix_cf(self, hidden_support, label_new_support, hidden_query, label_new_query, shuffle_list,
               shuffle_channel_id):

        hidden_support, label_new_support = self.channel_shuffle(hidden_support, label_new_support, shuffle_list,
                                                                 shuffle_channel_id)
        hidden_query, label_new_query = self.channel_shuffle(hidden_query, label_new_query, shuffle_list,
                                                             shuffle_channel_id)

        hidden_query, label_new_query, label_new_support, lam = self.mixup_data(hidden_support, label_new_support, hidden_query,
                                                        label_new_query)

        return hidden_support, label_new_support, hidden_query, label_new_query, lam

    def functional_forward_cf_mix_query(self, hidden_support, label_support, hidden_query, label_query, sel_layer,
                                                 shuffle_list, shuffle_channel_id, weights,
                                                 is_training=True):

        flag = 0

        for layer in range(4):
            if layer == sel_layer:
                hidden_support, label_new_support, hidden_query, label_new_query, lam = self.mix_cf(hidden_support,
                                                                                                    label_support,
                                                                                                    hidden_query,
                                                                                                    label_query,
                                                                                                    shuffle_list,
                                                                                                    shuffle_channel_id)
                flag = 1

            if not flag:
                hidden_support = self.functional_conv_block(hidden_support, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                                            weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                                            is_training)

            hidden_query = self.functional_conv_block(hidden_query, weights[f'net.{layer}.0.weight'], weights[f'net.{layer}.0.bias'],
                                                      weights.get(f'net.{layer}.1.weight'), weights.get(f'net.{layer}.1.bias'),
                                                      is_training)


        hidden_query = hidden_query.view(hidden_query.size(0), -1)

        x = F.linear(hidden_query, weights['logits.weight'], weights['logits.bias'])

        return x, label_new_query, label_new_support, lam
