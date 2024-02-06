import tensorflow as tf
import glob
import random
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def get_ds_info(ds_name):
    if ds_name == 'sleepedf':
        ds_info = {
            'ds_name': 'sleep',
            'class_size': 5,
            'win_size': 3000,
            'mod_name': ['EEG_Fpz_Cz', 'EEG_Pz_Oz', 'EOG_horizontal', 'EMG_submental'],
            'mod_dim': [1, 1, 1, 1],
            'labels' : ['1','2','3','4','5'],
            'num_device' : 4,
            'sensor_per_device' : [1,1,1,1]
        }
        return ds_info
    elif ds_name == 'wesad':
        ds_info = {
            'ds_name': 'wesad',
            'class_size': 4,
            'win_size': 1000,
            'mod_name': ['ECG','EMG','EDA'],
            'mod_dim': [1,1,1],
            'labels' : ['Baseline','Stress','Amusement','Meditation'],
            'num_device' : 3,
            'sensor_per_device': [1,1, 1]
        }
        return ds_info
    elif ds_name == 'pamap2':
        ds_info = {
            'ds_name': 'pamap2',
            'class_size': 12,
            'win_size': 512,
            'mod_name': ['HAND_ACC', 'HAND_GYRO', 'ANKLE_ACC', 'ANKLE_GYRO', 'CHEST_ACC', 'CHEST_GYRO'],
            'mod_dim': [3, 3, 3, 3, 3, 3],
            'labels' : ['lying', 'sitting','standing','walking','running','cycling','nordic_walking',
                        'ascending_stairs','descending_stairs','vaccuum_cleaning','ironing','rope_jumping'],
            'num_device' : 3,
            'sensor_per_device': [2,2,2]
        }
        return ds_info
    elif ds_name == 'pamap2_hr':
        ds_info = {
            'ds_name': 'pamap2_hr',
            'class_size': 12,
            'win_size': 512,
            'mod_name': ['HAND_ACC', 'HAND_GYRO', 'ANKLE_ACC', 'ANKLE_GYRO', 'CHEST_ACC', 'CHEST_GYRO','heartrate'],
            'mod_dim': [3, 3, 3, 3, 3, 3,1],
            'labels' : ['lying', 'sitting','standing','walking','running','cycling','nordic_walking',
                        'ascending_stairs','descending_stairs','vaccuum_cleaning','ironing','rope_jumping'],
            'num_device' : 4,
            'sensor_per_device': [2, 2, 2, 1]
        }
        return ds_info
    return 'Error >>>> Dataset is not recognized!'


def one_hot_encoder(in_array):
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(in_array)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    cls_onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return cls_onehot_encoded


# DATA LOADER:
def load_sleep4_ds(path, label_efficiency, mode="train", state="state"):
    fnames = glob.glob(os.path.join(path, "*.npz"))
    fnames.sort()
    fnames = np.asarray(fnames)
    # print("fnames : ",fnames)
    all_data = []
    all_label = []
    if mode=='vis':
        mode = 'train'
    if mode == "train":
        fnames = fnames[0:-15]
    if mode == "val":
        fnames = fnames[-15:-8]
    elif mode == "test":
        fnames = fnames[-8:]

    for i in range(len(fnames)):
        with np.load(fnames[i]) as data:
            x = data['x']
            y = data['y']
            # Add duplicated samples to handle imbalance label
            if mode in ["train","val"]:
                max_label = np.argmax(np.unique(y, return_counts=True)[1])
                x = np.vstack([x, x[np.where((y != max_label) & (y != 2))]])
                y = np.hstack([y, y[np.where((y != max_label) & (y != 2))]])
            all_data.append(x)#.transpose(0, 2, 1))
            all_label.append(y)

    all_data = np.vstack(all_data)
    all_label = np.hstack(all_label)

    sample_size = int(all_label.shape[0] * label_efficiency)
    all_data.astype('float32')
    idx = random.sample(range(0, all_label.shape[0]), sample_size)
    all_data = all_data[idx]
    all_label = all_label[idx]
    return all_data, one_hot_encoder(all_label) # ['EEG_Fpz_Cz', 'EEG_Pz_Oz', 'EOG_horizontal', 'EMG_submental']


def load_wesad_ds(path, label_efficiency, mode="train", state="state", held_out=0):
    all_users = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    user_list = all_users #[]
    """
    if mode == "test":
        user_list = [u for u in range(held_out, held_out+3)]
    elif mode == "val":
        user_list = [0,1] if held_out == 12 else [held_out+3, held_out+4]
    elif mode == "train":
        val_user = [0,1] if held_out == 12 else [held_out+3,held_out+4]
        user_list = [i for i in range(0, len(all_users)) if i not in [held_out,held_out+1,held_out+2, val_user[0], val_user[1]]]
    user_list = [all_users[i] for i in user_list]"""
    if mode == "val":
        mode = 'train'

    ext = '.npy'
    all_data, all_label = [], []
    for i in user_list:
        x = np.load(os.path.join(path, "x_" + str(i) + mode + ext))
        y = np.load(os.path.join(path, "y_" + str(i) + mode + ext))
        all_data.append(x.astype('float32'))
        all_label.append(y)

    all_data = np.vstack(all_data)
    all_label = np.vstack(all_label)

    sample_size = int(all_label.shape[0] * label_efficiency)
    if mode != 'test':
        sample_size = sample_size if sample_size > 64 else 64
    # print(" --------   {} data shape :  {}/{}".format(mode, sample_size, all_label.shape[0]))
    # all_data=all_data.transpose(0, 2, 1)
    # all_data = all_data.reshape((all_data.shape[0],6,3,all_data.shape[-1]))
    all_data.astype('float32')
    idx = random.sample(range(0, all_label.shape[0]), sample_size)
    all_data = all_data[idx]
    all_label = all_label[idx]

    return all_data, all_label#one_hot_encoder(all_label)

def load_wesad_heldout_ds(path, label_efficiency, mode="train", state="state", held_out=0):
    all_users = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    user_list = []

    if mode == "test":
        user_list = [u for u in range(held_out, held_out+3)]
    elif mode == "val":
        user_list = [0,1] if held_out == 12 else [held_out+3, held_out+4]
    elif mode == "train":
        val_user = [0,1] if held_out == 12 else [held_out+3,held_out+4]
        user_list = [i for i in range(0, len(all_users)) if i not in [held_out,held_out+1,held_out+2, val_user[0], val_user[1]]]
    user_list = [all_users[i] for i in user_list]

    ext = '.npy'
    all_data, all_label = [], []
    for i in user_list:
        x = np.load(os.path.join(path, "x_" + str(i) + "train" + ext))
        y = np.load(os.path.join(path, "y_" + str(i) + "train" + ext))
        all_data.append(x.astype('float32'))
        all_label.append(y)
        x = np.load(os.path.join(path, "x_" + str(i) + "test" + ext))
        y = np.load(os.path.join(path, "y_" + str(i) + "test" + ext))
        all_data.append(x.astype('float32'))
        all_label.append(y)

    all_data = np.vstack(all_data)
    all_label = np.vstack(all_label)

    sample_size = int(all_label.shape[0] * label_efficiency)
    if mode != 'test':
        sample_size = sample_size if sample_size > 64 else 64
    # print(" --------   {} data shape :  {}/{}".format(mode, sample_size, all_label.shape[0]))
    # all_data=all_data.transpose(0, 2, 1)
    # all_data = all_data.reshape((all_data.shape[0],6,3,all_data.shape[-1]))
    all_data.astype('float32')
    idx = random.sample(range(0, all_label.shape[0]), sample_size)
    all_data = all_data[idx]
    all_label = all_label[idx]

    return all_data, all_label#one_hot_encoder(all_label)
def load_pamap2_ds_held_out(path, label_efficiency=1.0, mode="train", held_out=0):
    user_list=[]
    if mode == "test":
        user_list.append(held_out)
    elif mode== "val":
        user_list.append(7 if held_out==0 else held_out-1 )
    elif mode == "train":
        val_user = 7 if held_out==0 else held_out-1
        user_list = [i for i in range(0,8) if i not in [held_out,val_user]]


    ext = '.npy'
    all_data=[]
    all_label=[]
    for i in user_list:
        all_data.append(np.load(os.path.join(path, 'X_' + str(i) + ext)))
        all_label.append(np.load(os.path.join(path, 'y_' + str(i) + ext)))

    all_data = np.vstack(all_data)
    all_label = np.vstack(all_label)

    sample_size = int(all_label.shape[0] * label_efficiency)
    if mode != 'test':
        sample_size = sample_size if sample_size > 64 else 64
    #print(" --------   {} data shape :  {}/{}".format(mode, sample_size, all_label.shape[0]))
    # all_data=all_data.transpose(0, 2, 1)
    # all_data = all_data.reshape((all_data.shape[0],6,3,all_data.shape[-1]))
    all_data.astype('float32')
    idx = random.sample(range(0, all_label.shape[0]), sample_size)
    all_data = all_data[idx]
    all_label = all_label[idx]
    # mn = np.min(np.min(all_data, axis=1), axis=0)
    # mx = np.max(np.max(all_data, axis=1), axis=0)
    # all_data = (all_data - mn) / (mx - mn)

    return all_data, all_label

def load_pamap2_ds(path, label_efficiency=1.0, mode="train", state="ssl"):
    if mode == 'vis':
        state = 'vis'
        mode = 'train'
    ext = '.npy'
    all_data = np.load(os.path.join(path, 'X_' + mode + ext))
    all_label = np.load(os.path.join(path, 'y_' + mode + ext))

    if state =="vis":
        x_val = np.load(os.path.join(path, 'X_val' + ext))
        all_data = np.vstack((all_data, x_val))
        y_val = np.load(os.path.join(path, 'y_val' + ext))
        all_label = np.vstack((all_label, y_val))

    sample_size = int(all_label.shape[0] * label_efficiency)
    #all_data=all_data.transpose(0, 2, 1)
    #all_data = all_data.reshape((all_data.shape[0],6,3,all_data.shape[-1]))
    all_data.astype('float32')
    idx = random.sample(range(0, all_label.shape[0]), sample_size)
    all_data = all_data[idx]
    all_label = all_label[idx]
    # mn = np.min(np.min(all_data, axis=1), axis=0)
    # mx = np.max(np.max(all_data, axis=1), axis=0)
    # all_data = (all_data - mn) / (mx - mn)

    return all_data, all_label


def get_data_bundle(name, data):
    if name == "sleepedf":
        return tf.data.Dataset.from_tensor_slices((data[:, :, [0]], data[:, :, [1]],
                                                   data[:, :, [2]], data[:, :, [3]]))

    if name == "pamap2":
        return tf.data.Dataset.from_tensor_slices((data[:, :, 0:3], data[:, :, 3:6],
                                                   data[:, :, 6:9], data[:, :, 9:12],
                                                   data[:, :, 12:15], data[:, :, 15:]))
    if name == "pamap2_hr":
        return tf.data.Dataset.from_tensor_slices((data[:, :, 0:3], data[:, :, 3:6],
                                                   data[:, :, 6:9], data[:, :, 9:12],
                                                   data[:, :, 12:15], data[:, :, 15:18],
                                                   data[:, :, -1]))
    if name == "wesad" or name=="wesad_overlap":
        return tf.data.Dataset.from_tensor_slices((data[:, :, [0]], data[:, :, [1]],
                                                   data[:, :, [2]]))

def load_dataset(path, ds_name, batch_size=8, mode="train", state="ssl", label_efficiency=1,held_out=0):
    if ds_name == "sleepedf":
        path = os.path.join(path, 'sleep')
        data, label = load_sleep4_ds(path, label_efficiency, mode=mode, state=state)
    if ds_name in ["pamap2","pamap2_hr"]:
        path = os.path.join(path, ds_name)
        data, label = load_pamap2_ds_held_out(path, label_efficiency, mode=mode, held_out=held_out)
    if ds_name == "wesad":
        path = os.path.join(path, ds_name)
        data, label = load_wesad_ds(path, label_efficiency, mode=mode, state=state, held_out=held_out)
    if ds_name == "wesad_overlap":
        path = os.path.join(path, "wesad")
        data, label = load_wesad_ds(path, label_efficiency, mode=mode, state=state, held_out=held_out)
    #print("{} , mode : {},  state:{}, shape: {}, label:{}".format(ds_name,mode, state, data.shape, label.shape))

    if state == 'ssl':
        trn_data = get_data_bundle(ds_name, data)
        trn_data = trn_data.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        return trn_data
    if ds_name=="wesad" and mode=="train":
        split_ratio = int(label.shape[0] * 0.8)
        trn_data = get_data_bundle(ds_name, data[0:split_ratio,:,:])
        trn_data = tf.data.Dataset.zip((trn_data, tf.data.Dataset.from_tensor_slices(label[0:split_ratio])))
        trn_data = trn_data.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

        val_data = get_data_bundle(ds_name, data[split_ratio:, :, :])
        val_data = tf.data.Dataset.zip((val_data, tf.data.Dataset.from_tensor_slices(label[split_ratio:])))
        val_data = val_data.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        return trn_data, val_data
    data = get_data_bundle(ds_name, data)

    data = tf.data.Dataset.zip((data, tf.data.Dataset.from_tensor_slices(label)))
    data = data.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    return data
