import json
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, sampler
import tqdm
import concurrent.futures
import pickle
import torch
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit import Chem

from utils.parser_utils import get_args

from utils import preprocess

class OurMetaDataset(Dataset):
    def __init__(self, args, target_assay):
        """
        A data provider class inheriting from Pytorch's Dataset class. It takes care of creating task sets for
        our few-shot learning model training and evaluation
        :param args: Arguments in the form of a Bunch object. Includes all hyperparameters necessary for the
        data-provider. For transparency and readability reasons to explicitly set as self.object_name all arguments
        required for the data provider, such that the reader knows exactly what is necessary for the data provider/
        """
        self.data_path = args.dataset_path
        self.compound_filename = args.compound_filename
        self.type_filename = args.type_filename
        self.fp_filename = args.fp_filename
        self.dataset_name = args.dataset_name
        self.fingerprint_dim = args.fingerprint_dim
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
        self.num_of_gpus = args.num_of_gpus
        self.batch_size = args.batch_size

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
        subfamilies = {}
        family_type = 0
        subfamily_type = 0
        assay_types = {}
        for cline in clines:
            cline = str(cline.strip())
            if 'assay_id' not in cline:
                strings = cline.split('\t')
                if strings[3] not in families:
                    families[strings[3]] = family_type
                    family_type += 1
                if strings[2] not in subfamilies:
                    subfamilies[strings[2]] = subfamily_type
                    subfamily_type += 1
                assay_types[int(strings[0])] = [families[strings[3]], subfamilies[strings[2]]]

        return assay_types


    def load_dataset(self):
        """
        Loads a dataset's dictionary files and splits the data according to the train_val_test_split variable stored
        in the args object.
        :return: Three sets, the training set, validation set and test sets (referred to as the meta-train,
        meta-val and meta-test in the paper)
        """
        rng = np.random.RandomState(seed=self.seed['val'])

        # here only split by training+testing and validation
        # to get the test set, just delete it from either training+testing or validation
        # while in the function
        experiment = preprocess.read_4276_txt(self.data_path, self.compound_filename)

        self.assay_ids = experiment.assays
        self.compounds = experiment.compounds
        self.assay_types = self.read_assay_type(self.type_filename)

        unique_families = []
        unique_subfamilies = []
        pic50_mean_tr = {}
        pic50_mean_te = {}

        for assay_id in tqdm.tqdm(self.assay_ids):
            mean = 0.0
            for idx, example in enumerate(experiment.training_set[assay_id]):
                if self.assay_types[example.assay_id][0] not in unique_families:
                    unique_families.append(self.assay_types[example.assay_id][0])
                if self.assay_types[example.assay_id][1] not in unique_subfamilies:
                    unique_subfamilies.append(self.assay_types[example.assay_id][1])
                mean += example.pic50_exp
            mean /= idx

            pic50_mean_tr[assay_id] = mean

            mean = 0.0

            for idx, example in enumerate(experiment.test_set[assay_id]):
                mean += example.pic50_exp
            mean /= idx

            pic50_mean_te[assay_id] = mean

        _, pic50_tr_bins = np.histogram(np.fromiter(pic50_mean_tr.values(), dtype=float), bins=3)
        _, pic50_te_bins = np.histogram(np.fromiter(pic50_mean_te.values(), dtype=float), bins=3)

        target_assays = [None] * 43
        groups = [None] * 43


        for assay_id in tqdm.tqdm(self.assay_ids):
            group = [self.assay_types[assay_id][0], self.assay_types[assay_id][1],
                     np.digitize(pic50_mean_tr[assay_id], pic50_tr_bins),
                     np.digitize(pic50_mean_te[assay_id], pic50_te_bins)]

            for i in range(43):
                if target_assays[i] is None:
                    target_assays[i] = []
                    target_assays[i].append(assay_id)
                    groups[i] = []
                    groups[i].append(group)
                    break
                elif len(target_assays[i]) < 101 and (group not in groups[i]):
                    target_assays[i].append(assay_id)
                    groups[i].append(group)
                    break
                else:
                    continue

            exist = False
            for i in range(43):
                if assay_id in target_assays[i]:
                    exist = True
                    break

            while not exist:
                i = np.random.randint(43)
                if len(target_assays[i]) < 101:
                    target_assays[i].append(assay_id)
                    groups[i].append(group)
                    exist = True



        print('test')



    def get_set(self, dataset_name, seed):
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
                                           size=self.batch_size, replace=False)
            rng.shuffle(selected_indices)

            for si in selected_indices:
                support_set_x.append(self.X_val[si]['support'])
                support_set_y.append(self.y_val[si]['support'])
                support_set_z.append(self.type_val[si]['support'])
                support_set_assay.append(self.assay_val[si])
                target_set_x.append(self.X_val[si]['target'])
                target_set_y.append(self.y_val[si]['target'])
                target_set_z.append(self.type_val[si]['target'])
                target_set_assay.append(self.assay_val[si])
        else:
            for bs in range(self.batch_size):
                support_set_x.append(self.X_test[0]['support'])
                support_set_y.append(self.y_test[0]['support'])
                support_set_z.append(self.type_test[0]['support'])
                support_set_assay.append(self.assay_test)
                target_set_x.append(self.X_test[0]['target'])
                target_set_y.append(self.y_test[0]['target'])
                target_set_z.append(self.type_test[0]['target'])
                target_set_assay.append(self.assay_test[0])


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
            self.get_set(self.current_set_name, seed=self.seed[self.current_set_name] + idx)

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
        self.num_of_gpus = args.num_of_gpus
        self.batch_size = args.batch_size
        self.samples_per_iter = args.samples_per_iter
        self.num_workers = args.num_dataprovider_workers
        self.total_train_iters_produced = 0
        self.dataset = OurMetaDataset(args=args, target_assay=target_assay)
        self.batches_per_iter = args.samples_per_iter
        self.full_data_length = self.dataset.data_length
        self.continue_from_iter(current_iter=current_iter)
        self.args = args

    def get_dataloader(self):
        """
        Returns a data loader with the correct set (train, val or test), continuing from the current iter.
        :return:
        """
        return DataLoader(self.dataset, batch_size=1, num_workers=self.num_workers, shuffle=False, drop_last=True)

    def continue_from_iter(self, current_iter):
        """
        Makes sure the data provider is aware of where we are in terms of training iterations in the experiment.
        :param current_iter:
        """
        self.total_train_iters_produced += (current_iter * (self.num_of_gpus * self.batch_size * self.samples_per_iter))

    def get_train_batches(self, total_batches=-1):
        """
        Returns a training batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param augment_images: Whether we want the images to be augmented.
        """
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length["train"] = total_batches #* self.dataset.batch_size
        self.dataset.switch_set(set_name="train", current_iter=self.total_train_iters_produced)
        self.total_train_iters_produced += (self.num_of_gpus * self.batch_size * self.samples_per_iter)
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
            self.dataset.data_length['val'] = total_batches #* self.dataset.batch_size
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
            self.dataset.data_length['test'] = total_batches #* self.dataset.batch_size
        self.dataset.switch_set(set_name='test')
        for sample_id, sample_batched in enumerate(self.get_dataloader()):
            yield sample_batched