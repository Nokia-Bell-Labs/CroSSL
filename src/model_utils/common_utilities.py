#!/usr/bin/env python3

import datetime
import os
import tensorflow as tf

from ds_utils.data_helper import get_ds_info

tf.random.set_seed(42)
tf.compat.v1.set_random_seed(42)

def setup_system(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for sel_gpu in gpus:
                tf.config.experimental.set_memory_growth(sel_gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)


    working_directory = os.path.join(args.output)
    os.makedirs(working_directory, exist_ok=True)
    print('working directory created at: ', working_directory)
    os.makedirs(os.path.join(working_directory, 'models/'), exist_ok=True)
    os.makedirs(os.path.join(working_directory, 'logs/'), exist_ok=True)
    os.makedirs(os.path.join(working_directory, 'results/'), exist_ok=True)
    os.makedirs(os.path.join(working_directory, 'plots/'), exist_ok=True)

    exp_name = args.mode + "_" + args.encoder + "_" + str(args.code_size) + "_" + str(args.proj_size) + "_" + str(args.batch_size)
    print(args)
    log_dir = os.path.join(working_directory, "logs",
                           datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + exp_name)
    # Encoder name
    encoder_name = 'encoder_' + args.encoder + '_' + args.dataset + '_c' + str(args.code_size) +\
                   "_b" + str(args.batch_size) + "_l" + str(args.lr_ssl) + "_cov" + str(args.coverage)

    # get dataset related information
    ds_info = get_ds_info(args.dataset)
    return working_directory, encoder_name, log_dir, ds_info
