from matplotlib.pyplot import psd
import numpy as np
import pickle
import os
import pdb
import random
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import tensorflow as tf
import argparse


class DictMap(dict):
    """
    Supports python dictionary dot notation with existing bracket notation
    Example:
    m = DictMap({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    m.first_name == m['first_name']
    """

    def __init__(self, *args, **kwargs):
        super(DictMap, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DictMap, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DictMap, self).__delitem__(key)
        del self.__dict__[key]


class Data(object):
    """
    Creates tf.data.Dataset object for each device
    Note: words device and position are used interchangeably
    """

    def __init__(self, path='/mnt/data/gsl', dataset_name='pamap2adl-2.0-0.0.dat', load_path=None, held_out=None):
        super(Data, self).__init__()
        self.path = path
        self.dataset_name = dataset_name
        if "realworld" in dataset_name:
            self.norm = 30.0
        elif "pamap2" in dataset_name:
            self.norm = 15.0
        elif "hhar" in dataset_name:
            self.norm = 30.0
        else:
            self.norm = 1.0

        self.info, self.training, self.test = self.load_dataset()

        print(self.norm, self.info)

        print("Generate Numpy Datasets")
        self.device_user_train = self.flatten_sessions(self.training)
        self.device_user_test = self.flatten_sessions(self.test)
        self.device_user_ds = self.combine_sessions(self.device_user_train, self.device_user_test)
        sample_set = next(iter(next(iter(self.device_user_ds.values())).values()))
        self.input_shape = sample_set[0].shape[1:]
        self.info['device_list'] = list(self.device_user_ds.keys())
        self.info['user_list'] = list(self.device_user_ds[self.info['device_list'][0]].keys())
        # Support .shape
    def get_train_data(self):
        return self.training

    def get_test_data(self):
        return self.test

    def load_dataset(self):
        filename = os.path.join(self.path, self.dataset_name)
        print(f"loading dataset from path {filename}")
        # pickle file (.dat) file contains a 3-tuple: info, training and testing
        # info: a dictionary of 'device_list', 'user_list' and 'session_list' for realworld-3.0-0.0
        # training: a dictionary (mapping from device in device_list to a dictionary
        #     (mapping from user name in user_list to a dictionary
        #         (mapping from session in session_list to a list of length 2, the first beging a ? x window_size (150) x channels (6) array, the second being an flat array of length ? containing the labels)
        #     )
        # )
        # testig is similar
        f = open(filename, 'rb')
        obj = pickle.load(f)
        f.close()
        return obj

    def flatten(self, data, mode='train', held_out=None):
        X = None
        y = None
        user_id = None
        # Ian: Potential unintended behaviour
        # the iteration for data, data[device] and data[device][user] are all unsorted
        # which might lead to different orders of things being traversed
        # Also, the check on user_index being inside self.held_out relies on the order of traversal of a dictionary
        # which might not be properly defined
        # After checks, it seems to be working ok for realworld dataset
        for device in data:
            for user_index, user in enumerate(data[device]):
                if held_out is not None:
                    # if it is a training set, exclude users which are held out
                    if mode == 'train' and user_index in held_out:
                        continue
                    # if it is a testing set, only include users in held out
                    elif mode == 'test' and not user_index in held_out:
                        continue

                for session in sorted(data[device][user].keys()):

                    XX = data[device][user][session][0]
                    yy = data[device][user][session][1]
                    # print(device, user, session, XX.shape, yy.shape)

                    if len(XX) == 0:
                        continue

                    X = XX if X is None else np.vstack((X, XX))
                    y = yy if y is None else np.hstack((y, yy))

                    user_id = np.repeat(user_index, yy.shape) if user_id is None else np.hstack(
                        (user_id, np.repeat(user_index, yy.shape)))

                    # Ian: Small test guaranteeing the lengths of the datasets are consistent
                    # assert(len(X) == len(y))
                    # assert(len(y) == len(user_id))
        X = X / self.norm  # norm is set as a constant = 30.0
        X = X.astype(np.float32)
        return X, y, user_id

    def get_position_dataset(self, position_name, held_out):
        position_train = self.flatten(
            {position_name: self.training[position_name]}, mode='train', held_out=held_out)

        # take the entire dataset from held_out user
        position_test = self.flatten({position_name: self.training[position_name]}, mode='test', held_out=held_out)

        return position_train, position_test

    def backup_current_dataset_references(self):
        self.backup_dataset_references = (DictMap(self.ds_train), DictMap(self.ds_test))

    def restore_dataset_references(self):
        train, test = self.backup_dataset_references
        self.ds_train, self.ds_test = (DictMap(train), DictMap(test))

    def flatten_sessions(self, data):

        new_dataset = {}
        for device in data:
            new_dataset[device] = {}
            for user in data[device]:
                X = []
                y = []
                for session in sorted(data[device][user].keys()):

                    XX = data[device][user][session][0]
                    yy = data[device][user][session][1]

                    if len(XX) == 0:
                        continue
                    X.append(XX)
                    y.append(yy)

                if len(X) == 0:
                    print(device, user)
                    continue
                X = np.concatenate(X, axis=0).astype(np.float32)
                X = X / self.norm  # norm is set as a constant = 30.0
                y = np.concatenate(y, axis=0)
                if 0 not in np.unique(y):
                    y = y - 1
                new_dataset[device][user] = (X, y)
        return new_dataset

    def combine_sessions(self, train_data, test_data):
        new_dataset = {}
        for device in train_data.keys():
            new_dataset[device] = {}
            common_users = sorted(list(set.intersection(set(train_data[device].keys()), set(test_data[device].keys()))))
            for user in common_users:
                X = np.concatenate((train_data[device][user][0], test_data[device][user][0]), axis=0)
                y = np.concatenate((train_data[device][user][1], test_data[device][user][1]), axis=0)
                new_dataset[device][user] = (X, y)
        return new_dataset


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    parser = argparse.ArgumentParser(
        description='Inputs to data loading script')
    parser.add_argument('--dataset_path', default='/mnt/data/gsl',
                        help='path of the dataset .dat file')
    parser.add_argument('--dataset_name', default='realworld-3.0-0.0.dat', choices=[
        'realworld-3.0-0.0.dat', 'opportunity-1.0-0.0.dat'], help='name of dataset file')
    parser.add_argument('--load_path', default=None,
                        help='path of the dataset TFrecords files')

    args = parser.parse_args()
    print(args.dataset_path, args.dataset_name)
    data = Data(args.dataset_path, args.dataset_name)
    import pdb

    pdb.set_trace()

    # walking = get_test_dataset(positions='head', activity='walking')
    # print(walking)