#!/usr/bin/env python3

import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='argument setting of the model')

    # hyperparameter
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of training')
    parser.add_argument('--epoch', type=int, default=50, help='number of training epochs')
    parser.add_argument('--lr_ssl', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr_cls', type=float, default=1e-4, help='learning rate for linear classifier')
    parser.add_argument('--optim_ssl', type=str, default='adam', help='optimizer for self supervised part')
    parser.add_argument('--optim_cls', type=str, default='adam', help='optimizer for classification part')
    parser.add_argument('--filters', type=int, default=64, help='Number of filters for the encoders')
    parser.add_argument('--scheduler', type=bool, default=True, help='if or not to use a scheduler')
    parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--datapath', type=str, default='../data', help='[ ../data ] prefix path to data directory')
    parser.add_argument('--output', type=str, default='../output_rev1', help='[ ../output ] prefix path to output directory')
    parser.add_argument('--label_eff', type=float, default=1.0, help='Label efficiency ratio for fine tuning')
    parser.add_argument('--masking', type=str, default='spatial', help='Masking strategy')
    parser.add_argument('--ds_encoder_only', type=bool, default=False, help='Using pretrained encoder only or encoder+aggregator in downstream task. ')
    # framework
    parser.add_argument('--criterion', type=str, default='cos_sim', choices=['cos_sim', 'NTXent'],
                        help='type of loss function for contrastive learning')
    parser.add_argument('--code_size', type=int, default=96, help='size of encoded features')
    parser.add_argument('--proj_size', type=int, default=32, help='size of projected features')
    parser.add_argument('--encoder', type=str, default='TCN', choices=['TCN', 'CAE', 'FCN'], help='type of encoder')
    parser.add_argument('--coverage', type=float, default=0.9,
                        help='ratio of modalities to be considered for the projector')
    parser.add_argument('--model_name', type=str, default='TCN', help='type of encoder')
    parser.add_argument('--exp_name', type=str, default='test', help='name of the experiment')
    parser.add_argument('--mode', type=str, default='ssl', help='mode of training [fixed | fine]')
    parser.add_argument('--load_encoder', type=bool, default=True,
                        help='if or not to use a existing pretrained encoder')
    parser.add_argument('--verbosity', type=int, default=0, help='set verbosity')
    parser.add_argument('--held_out', type=int, default=0, help='held out user for test')
    # GPU setup
    parser.add_argument('--cuda', default=0, type=int, help='cuda device ID，0/1')
    parser.add_argument('--gpu_device', type=str, default='7', help='specify the GPU')
    parser.add_argument('--n_process_per_gpu', default=1, type=int,
                        help='number of processes per gpu (for multi-processing only)')
    parser.add_argument('--missing_ft', type=bool, default=False, help='If True then fine_tuning is done with missing devices.')
    # Random seed for reproducibility
    parser.add_argument('--random_seed', type=int, default=42)


    # dataset
    parser.add_argument('--dataset', type=str,
                        choices=['ucihar', 'shar', 'hhar', 'sleepedf', 'wesad', 'pamap2', 'pamap2_hr'],
                        help='name of dataset')

    # augmentation
    parser.add_argument('--aug', type=bool, default=False, help='if or not applying augmentation')

    # log
    parser.add_argument('--logdir', type=str, default='log/', help='log directory')
    parser.add_argument('--info', type=str, default='', help='additional info added to the logs folder name')

    # loss
    parser.add_argument('--loss', type=str, default='vicreg', help='loss function [vicreg1 | vicreg2 | mse]')
    parser.add_argument("--sim-coeff", type=float, default=10.0, help='Invariance regularization loss coefficient')
    parser.add_argument("--std-coeff", type=float, default=10.0, help='Variance regularization loss coefficient')
    parser.add_argument("--cov-coeff", type=float, default=100.0, help='Covariance regularization loss coefficient')
    parser.add_argument("--temp", type=float, default=0.2, help='Temperature for scaling the similarities')

    # plot
    parser.add_argument('--plt', type=bool, default=False, help='if or not to plot results')
    parser.add_argument('--tsne', type=bool, default=False, help='[0|1] Visualize the representations using t-SNE')

    return parser