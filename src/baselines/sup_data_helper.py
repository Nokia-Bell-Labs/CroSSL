import json

import numpy as np
import glob
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from ds_utils.data_helper import load_sleep4_ds, load_pamap2_ds_held_out



def one_hot_encoder(in_array):
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(in_array)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    cls_onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return cls_onehot_encoded

def load_har_ds_sup(path, mode="train"):

    path = os.path.join(path,"processed")
    acc = np.load(os.path.join(path,"act_"+mode+"_acc2.npy"))
    gyro = np.load(os.path.join(path, "act_" + mode + "_gyr2.npy"))
    label = np.load(os.path.join(path, "act_" + mode + "_y2.npy"))

    data = np.concatenate((acc, gyro),axis=-1)
    print(data.shape)
    return data, label

def load_sleep4_ds_sup(path, mode="train"):
    fnames = glob.glob(os.path.join(path, "*.npz"))
    fnames.sort()
    fnames = np.asarray(fnames)
    # print("fnames : ",fnames)
    all_data = []
    all_label = []
    if mode == "train":
        fnames = fnames[0:22]
    elif mode == "test":
        fnames = fnames[31:]

    for i in range(len(fnames)):
        with np.load(fnames[i]) as data:
            x = data['x']
            y = data['y']
            # Add duplicated samples to handle imbalance label
            if mode == "train":
                max_label = np.argmax(np.unique(y, return_counts=True)[1])
                x = np.vstack([x, x[np.where((y != max_label) & (y != 2))]])
                y = np.hstack([y, y[np.where((y != max_label) & (y != 2))]])
            all_data.append(x)
            all_label.append(y)

    all_data = np.vstack(all_data)
    all_label = np.hstack(all_label)
    print("sleepedf data: ", all_data.shape)
    return all_data, one_hot_encoder(all_label) # ['EEG_Fpz_Cz', 'EEG_Pz_Oz', 'EOG_horizontal', 'EMG_submental']

def load_pamap2_ds_sup(path, mode="train"):
    ext = '.npy'

    x = np.load(os.path.join(path, 'X_'+mode + ext))
    y = np.load(os.path.join(path, 'y_'+mode + ext))

    x.astype('float32')

    # mn = np.min(np.min(x, axis=1), axis=0)
    # mx = np.max(np.max(x, axis=1), axis=0)
    # x = (x - mn) / (mx - mn)

    return x,y


def load_opportunity_ds_sup(path, mode="train"):
    ext = '.npy'
    all_data = np.load(os.path.join(path, 'X_' + mode + ext))
    all_label = np.load(os.path.join(path, 'y_' + mode + ext))

    # all_data=all_data.transpose(0, 2, 1)

    all_data.astype('float32')
    all_label = one_hot_encoder(all_label)
    print("opportunity : ", all_data.shape)

    return all_data, all_label

def load_wesad_ds_sup(path, mode='train'):
    ext = '.npy'
    all_data, all_label = [], []
    for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]:
        x = np.load(os.path.join(path, "x_" + str(i) + mode + ext))
        y = np.load(os.path.join(path, "y_" + str(i) + mode + ext))
        all_data.append(x.astype('float32'))
        all_label.append(y)
    all_data = np.vstack(all_data)
    all_label = np.vstack(all_label)

    all_data.astype('float32')
    # all_label = one_hot_encoder(all_label)

    print("wesad : ", all_data.shape)
    return all_data, all_label


def load_sup_dataset(path, ds_name, batch_size=8, mode="train", state="ssl", label_efficiency=1,held_out=0):
    if ds_name == "sleepedf":
        path = os.path.join(path, 'sleep')
        data, label = load_sleep4_ds(path, label_efficiency, mode=mode, state=state)
    if ds_name == "pamap2":
        path = os.path.join(path, 'pamap2')
        data, label = load_pamap2_ds_held_out(path, label_efficiency, mode=mode, held_out=held_out)
    return data, label