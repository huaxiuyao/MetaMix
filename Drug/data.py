import json
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, sampler
import tqdm
import concurrent.futures
import pickle
import torch
import preprocess
import copy


class OurMetaDataset(Dataset):
    def __init__(self, args, target_assay):
        """
        A data provider class inheriting from Pytorch's Dataset class. It takes care of creating task sets for
        our few-shot learning model training and evaluation
        :param args: Arguments in the form of a Bunch object. Includes all hyperparameters necessary for the
        data-provider. For transparency and readability reasons to explicitly set as self.object_name all arguments
        required for the data provider, such that the reader knows exactly what is necessary for the data provider/
        """
        self.data_path = os.path.join(args.datadir, args.dataset_path)
        self.compound_filename = os.path.join(args.datadir, args.compound_filename)
        self.type_filename = os.path.join(args.datadir, args.type_filename)
        self.fp_filename = os.path.join(args.datadir, args.fp_filename)
        self.dataset_name = args.dataset_name
        self.fingerprint_dim = args.dim_w
        self.args = args
        self.train_val_split = args.train_val_split
        self.current_set_name = "train"
        val_rng = np.random.RandomState(seed=args.val_seed)
        val_seed = val_rng.randint(1, 999999)
        train_rng = np.random.RandomState(seed=args.train_seed)
        train_seed = train_rng.randint(1, 999999)
        test_rng = np.random.RandomState(seed=args.val_seed)
        test_seed = test_rng.randint(1, 999999)
        args.val_seed = val_seed
        args.train_seed = train_seed
        args.test_seed = test_seed

        self.init_seed = {"train": args.train_seed, "val": args.val_seed, 'test': args.val_seed}
        self.seed = {"train": args.train_seed, "val": args.val_seed, 'test': args.val_seed}
        self.batch_size = args.meta_batch_size

        self.num_evaluation_tasks = args.num_evaluation_tasks

        self.train_index = 0
        self.val_index = 0
        self.test_index = 0

        self.target_assay = target_assay

        self.rng = np.random.RandomState(seed=self.seed['val'])

        self.load_dataset()

        print("data", self.data_length)
        self.observed_seed_set = None

    def read_assay_type(self, filepath):
        type_file = open(filepath, 'r', encoding='UTF-8', errors='ignore')
        clines = type_file.readlines()
        type_file.close()

        families = {}
        family_type = 0
        assay_types = {}
        for cline in clines:
            cline = str(cline.strip())
            if 'assay_id' not in cline:
                strings = cline.split('\t')
                if strings[3] not in families:
                    families[strings[3]] = family_type
                    family_type += 1
                assay_types[int(strings[0])] = families[strings[3]]

        return assay_types

    def load_dataset(self):
        """
        Loads a dataset's dictionary files and splits the data according to the train_val_test_split variable stored
        in the args object.
        :return: Three sets, the training set, validation set and test sets (referred to as the meta-train,
        meta-val and meta-test in the paper)
        """

        self.split_name_train_val_test = pickle.load(
            open(os.path.join(self.args.datadir, 'drug_split_id_group{}.pickle'.format(self.args.drug_group)), 'rb'))

        rng = np.random.RandomState(seed=self.seed['val'])

        # here only split by training+testing and validation
        # to get the test set, just delete it from either training+testing or validation
        # while in the function
        experiment = preprocess.read_4276_txt(self.data_path, self.compound_filename)
        # preprocess.save_fp(self.compound_filename, self.fp_filename)

        self.assay_ids = experiment.assays
        self.compounds = experiment.compounds
        self.assay_types = self.read_assay_type(self.type_filename)
        self.compound_fp = np.load(self.fp_filename, allow_pickle=True)

        support_set = experiment.training_set
        target_set = experiment.test_set

        self.n_assays = len(support_set)

        self.indices = []
        shuffled_assay_ids = copy.deepcopy(self.assay_ids)
        rng.shuffle(shuffled_assay_ids)

        train_id, val_id, test_id = int(self.train_val_split[0] * self.n_assays), int(
            (self.train_val_split[0] + self.train_val_split[1]) * self.n_assays), int(self.n_assays)

        print(train_id, val_id)

        self.X_train = [None] * 4100
        self.y_train = [None] * 4100
        self.type_train = [None] * 4100
        self.assay_train = [None] * 4100

        self.X_val = [None] * 76
        self.y_val = [None] * 76
        self.type_val = [None] * 76
        self.assay_val = [None] * 76

        self.X_test = [None] * 100
        self.y_test = [None] * 100
        self.type_test = [None] * 100
        self.assay_test = [None] * 100

        self.data_length = {}
        self.data_length['train'] = 4100
        self.data_length['val'] = 76
        self.data_length['test'] = 100
        self.train_indices = []
        self.val_indices = []
        self.test_indices = []

        ma_support, ma_target = 0, 0

        train_cnt, val_cnt, test_cnt = 0, 0, 0

        for idx, assay_id in tqdm.tqdm(enumerate(shuffled_assay_ids)):
            X_supp_tgt = {}
            y_supp_tgt = {}
            assay_supp_tgt = {}

            x_tmp = np.zeros((len(support_set[assay_id]), self.fingerprint_dim), dtype=np.float32)
            y_tmp = np.zeros((len(support_set[assay_id]),), dtype=np.float32)
            z_tmp = np.zeros((len(support_set[assay_id]), 2), dtype=np.int32)
            count_exp = 0

            for example in support_set[assay_id]:
                # smile = self.compounds[example.compound_id]
                # ml = Chem.MolFromSmiles(smile)
                # fp = AllChem.GetMorganFingerprintAsBitVect(ml, 2, nBits=self.fingerprint_dim)
                # tmp_array = np.zeros((0,), dtype=np.int32)
                # DataStructs.ConvertToNumpyArray(fp, tmp_array)
                x_tmp[count_exp] = self.compound_fp.item().get(example.compound_id)
                y_tmp[count_exp] = example.pic50_exp
                z_tmp[count_exp] = np.array(
                    [self.assay_ids.index(example.assay_id), self.assay_types[example.assay_id]])
                count_exp += 1

            # x_tmp[np.isnan(x_tmp)] = 0
            # x_tmp = x_tmp / np.linalg.norm(x_tmp, axis=1, keepdims=True)
            X_supp_tgt['support'] = x_tmp
            y_supp_tgt['support'] = y_tmp
            assay_supp_tgt['support'] = z_tmp

            x_tmp = np.zeros((len(target_set[assay_id]), self.fingerprint_dim), dtype=np.float32)
            y_tmp = np.zeros((len(target_set[assay_id]),), dtype=np.float32)
            z_tmp = np.zeros((len(target_set[assay_id]), 2), dtype=np.int32)
            count_exp = 0

            for example in target_set[assay_id]:
                # smile = self.compounds[example.compound_id]
                # ml = Chem.MolFromSmiles(smile)
                # try:
                #     fp = AllChem.GetMorganFingerprintAsBitVect(ml, 2, nBits=self.fingerprint_dim)
                # except:
                #     try:
                #         smile = self.compounds['DB00014']
                #     except:
                #         smile = self.compounds['Favipiravir']
                #
                #     ml = Chem.MolFromSmiles(smile)
                #     fp = AllChem.GetMorganFingerprintAsBitVect(ml, 2, nBits=self.fingerprint_dim)
                # tmp_array = np.zeros((0,), dtype=np.int32)
                # DataStructs.ConvertToNumpyArray(fp, tmp_array)
                x_tmp[count_exp] = self.compound_fp.item().get(example.compound_id)
                y_tmp[count_exp] = example.pic50_exp
                z_tmp[count_exp] = np.array(
                    [self.assay_ids.index(example.assay_id), self.assay_types[example.assay_id]])
                count_exp += 1

            # x_tmp[np.isnan(x_tmp)] = 0
            # x_tmp = x_tmp / np.linalg.norm(x_tmp, axis=1, keepdims=True)
            X_supp_tgt['target'] = x_tmp
            y_supp_tgt['target'] = y_tmp
            assay_supp_tgt['target'] = z_tmp

            ma_support = max(ma_support, X_supp_tgt['support'].shape[0])
            ma_target = max(ma_target, X_supp_tgt['target'].shape[0])

            ####### for debug
            # if idx > 9:
            #     break

            # here split the training and the validation part
            if assay_id in self.split_name_train_val_test['train']:
                self.X_train[train_cnt] = X_supp_tgt
                self.y_train[train_cnt] = y_supp_tgt
                self.type_train[train_cnt] = assay_supp_tgt
                self.assay_train[train_cnt] = assay_id
                self.train_indices.append(train_cnt)
                train_cnt += 1
            elif assay_id in self.split_name_train_val_test['val']:
                self.X_val[val_cnt] = X_supp_tgt
                self.y_val[val_cnt] = y_supp_tgt
                self.type_val[val_cnt] = assay_supp_tgt
                self.assay_val[val_cnt] = assay_id
                self.val_indices.append(val_cnt)
                val_cnt += 1
            elif assay_id in self.split_name_train_val_test['test']:
                self.X_test[test_cnt] = X_supp_tgt
                self.y_test[test_cnt] = y_supp_tgt
                self.type_test[test_cnt] = assay_supp_tgt
                self.assay_test[test_cnt] = assay_id
                self.test_indices.append(test_cnt)
                test_cnt += 1
            else:
                print(assay_id)

        print(ma_support, ma_target, train_cnt, val_cnt, test_cnt)

    def get_set(self, dataset_name, seed, idx):
        rng = np.random.RandomState(seed)

        support_set_x = []
        support_set_y = []
        support_set_z = []
        support_set_assay = []
        target_set_x = []
        target_set_y = []
        target_set_z = []
        target_set_assay = []

        if dataset_name == 'train':
            selected_indices = rng.choice(self.train_indices,
                                          size=self.batch_size, replace=False)
            rng.shuffle(selected_indices)

            for si in selected_indices:
                # print(self.type_train[si]['support'][0])
                support_set_x.append(self.X_train[si]['support'])
                support_set_y.append(self.y_train[si]['support'])
                support_set_z.append(self.type_train[si]['support'])
                support_set_assay.append(self.assay_train[si])
                target_set_x.append(self.X_train[si]['target'])
                target_set_y.append(self.y_train[si]['target'])
                target_set_z.append(self.type_train[si]['target'])
                target_set_assay.append(self.assay_train[si])

        elif dataset_name == 'val':
            selected_indices = rng.choice(self.val_indices,
                                          # size=self.batch_size, replace=False)
                                          size=len(self.val_indices), replace=False)
            rng.shuffle(selected_indices)

            for si in selected_indices:
                # print(self.type_val[si]['support'][0])
                support_set_x.append(self.X_val[si]['support'])
                support_set_y.append(self.y_val[si]['support'])
                support_set_z.append(self.type_val[si]['support'])
                support_set_assay.append(self.assay_val[si])
                target_set_x.append(self.X_val[si]['target'])
                target_set_y.append(self.y_val[si]['target'])
                target_set_z.append(self.type_val[si]['target'])
                target_set_assay.append(self.assay_val[si])
        elif dataset_name == 'test':
            selected_indices = idx
            support_set_x.append(self.X_test[selected_indices]['support'])
            support_set_y.append(self.y_test[selected_indices]['support'])
            support_set_z.append(self.type_test[selected_indices]['support'])
            support_set_assay.append(self.assay_test[selected_indices])
            target_set_x.append(self.X_test[selected_indices]['target'])
            target_set_y.append(self.y_test[selected_indices]['target'])
            target_set_z.append(self.type_test[selected_indices]['target'])
            target_set_assay.append(self.assay_test[selected_indices])

        return support_set_x, support_set_y, support_set_z, support_set_assay, \
               target_set_x, target_set_y, target_set_z, target_set_assay, seed

    def __len__(self):
        total_samples = self.data_length[self.current_set_name]
        return total_samples

    def length(self, set_name):
        self.switch_set(set_name=set_name)
        return len(self)

    def switch_set(self, set_name, current_iter=0):
        self.current_set_name = set_name
        if set_name == "train":
            self.update_seed(dataset_name=set_name, seed=self.init_seed[set_name] + current_iter)
            ## shuffle the datasets for training

    def update_seed(self, dataset_name, seed=100):
        self.seed[dataset_name] = seed

    def __getitem__(self, idx):
        support_set_x, support_set_y, support_set_z, support_set_assay, \
        target_set_x, target_set_y, target_set_z, target_set_assay, seed = \
            self.get_set(self.current_set_name, seed=self.seed[self.current_set_name] + idx, idx=idx)

        return support_set_x, support_set_y, support_set_z, support_set_assay, \
               target_set_x, target_set_y, target_set_z, target_set_assay, seed


    def reset_seed(self):
        self.seed = self.init_seed


class MetaLearningSystemDataLoader(object):
    def __init__(self, args, current_iter=0, target_assay=None):
        """
        Initializes a meta learning system dataloader. The data loader uses the Pytorch DataLoader class to parallelize
        batch sampling and preprocessing.
        :param args: An arguments NamedTuple containing all the required arguments.
        :param current_iter: Current iter of experiment. Is used to make sure the data loader continues where it left
        of previously.
        """
        self.batch_size = args.meta_batch_size
        self.total_train_iters_produced = 0
        self.dataset = OurMetaDataset(args, target_assay=target_assay)
        self.full_data_length = self.dataset.data_length
        self.continue_from_iter(current_iter=current_iter)

    def get_dataloader(self):
        """
        Returns a data loader with the correct set (train, val or test), continuing from the current iter.
        :return:
        """
        return DataLoader(self.dataset, batch_size=1, num_workers=1, shuffle=False, drop_last=True)

    def continue_from_iter(self, current_iter):
        """
        Makes sure the data provider is aware of where we are in terms of training iterations in the experiment.
        :param current_iter:
        """
        self.total_train_iters_produced += (current_iter * (self.batch_size))

    def get_train_batches(self, total_batches=-1):
        """
        Returns a training batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param augment_images: Whether we want the images to be augmented.
        """
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length["train"] = total_batches  # * self.dataset.batch_size
        self.dataset.switch_set(set_name="train", current_iter=self.total_train_iters_produced)
        self.total_train_iters_produced += self.batch_size
        for sample_id, sample_batched in enumerate(self.get_dataloader()):
            yield sample_batched

    def get_val_batches(self, total_batches=-1):
        """
        Returns a validation batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param augment_images: Whether we want the images to be augmented.
        """
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length['val'] = total_batches  # * self.dataset.batch_size
        self.dataset.switch_set(set_name="val")
        for sample_id, sample_batched in enumerate(self.get_dataloader()):
            yield sample_batched

    def get_test_batches(self, total_batches=-1):
        """
        Returns a testing batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param augment_images: Whether we want the images to be augmented.
        """
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length['test'] = total_batches  # * self.dataset.batch_size
        self.dataset.switch_set(set_name='test')
        for sample_id, sample_batched in enumerate(self.get_dataloader()):
            yield sample_batched