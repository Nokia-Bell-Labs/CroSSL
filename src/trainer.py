import os
import model_utils.loss as loss
import matplotlib.pyplot as plt

from evaluation.custom_metrics import CustomMetrics
from model_utils.encoders import *
from ds_utils.data_helper import load_dataset, get_ds_info
from model_utils.model import cm_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# gpu = len(tf.config.list_physical_devices('GPU')) > 0
# print("GPU is", "available" if gpu else "NOT AVAILABLE")


def plot_training(H, working_dir, exp_name):
    with plt.xkcd():
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle(exp_name)
        ax1.plot(H.history["loss"], label="train_loss")
        ax1.plot(H.history["val_loss"], label="val_loss")
        ax1.set_title("Training Loss")
        ax1.set(xlabel='Epoch', ylabel='loss')
        ax2.plot(H.history["categorical_accuracy"], label="train_acc")
        ax2.plot(H.history["val_categorical_accuracy"], label="val_acc")
        ax2.set_title("Training Accuracy")
        ax2.set(xlabel='Epoch', ylabel='Accuracy')
        handles, labels = ax2.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center')
        # plt.show()
        plt.savefig(os.path.join(working_dir, "plots", "LRCURVE_" + exp_name + ".png"))


def setup_optimizer(optim, lr):
    if optim == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr)
    elif optim == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(lr)
    elif optim == 'sgd':
        optimizer = tf.keras.optimizers.SGD(lr)
    return optimizer


def setup_ssl_training(log_dir, args):
    tensorboard_callback_ssl = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    ssl_es = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5, verbose=args.verbosity,
                                              restore_best_weights=True)
    cbs = [tensorboard_callback_ssl, ssl_es]
    # Optimizer
    optimizer_ssl = setup_optimizer(args.optim_ssl, args.lr_ssl)
    custom_loss_obj = loss.CustomLoss(temperature=args.temp, sim_coeff=args.sim_coeff,
                                      std_coeff=args.std_coeff, cov_coeff=args.cov_coeff)
    loss_fn = custom_loss_obj.get_loss_fn(args.loss)
    # set up training, validation, and test set
    ssl_data = load_dataset(args.datapath, args.dataset, batch_size=args.batch_size, mode="train", state='ssl',
                            label_efficiency=1, held_out=args.held_out)

    return cbs, optimizer_ssl, loss_fn, ssl_data


def setup_agg_training(log_dir, args):
    tensorboard_callback_ssl = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    es_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5, verbose=args.verbosity,
                                                   restore_best_weights=True)
    # lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    cbs = [tensorboard_callback_ssl, es_callback]
    # Loss function
    custom_loss_obj = loss.CustomLoss(temperature=args.temp, sim_coeff=args.sim_coeff,
                                      std_coeff=args.std_coeff, cov_coeff=args.cov_coeff)
    loss_fn = custom_loss_obj.get_loss_fn(args.loss)
    optimizer_ag = setup_optimizer(args.optim_ssl, args.lr_ssl)
    return cbs, optimizer_ag, loss_fn


def setup_cls_training(log_dir, args):
    tensorboard_callback_cls = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # Early Stopping to prevent overfitting
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=args.verbosity,
                                          restore_best_weights=True)
    # Check point
    # mcp_save = tf.keras.callbacks.ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    cbs = [tensorboard_callback_cls, es]
    # Optimizer
    optimizer_cls = setup_optimizer(args.optim_cls, args.lr_cls)
    # metrics
    metrics = [tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"),
               tf.keras.metrics.AUC(name="auc"), CustomMetrics.f1_m],
    # WandbCallback.init(project="CMSSRL", id="linear-eval-relu-last") #classifier_last_dense
    if args.dataset=="wesad":
        trn_data,val_data = load_dataset(path=args.datapath,
                                ds_name=args.dataset,
                                batch_size=args.batch_size,
                                mode="train",
                                state="fine",
                                label_efficiency=args.label_eff,
                                held_out=args.held_out)
    else:
        trn_data = load_dataset(path=args.datapath,
                                ds_name=args.dataset,
                                batch_size=args.batch_size,
                                mode="train",
                                state="fine",
                                label_efficiency=args.label_eff,
                                held_out=args.held_out)

        val_data = load_dataset(path=args.datapath,
                                ds_name=args.dataset,
                                batch_size=args.batch_size,
                                mode="val",
                                state="fine",
                                label_efficiency=args.label_eff,
                                held_out=args.held_out)

    tst_data = load_dataset(path=args.datapath,
                            ds_name=args.dataset,
                            batch_size=args.batch_size,
                            mode="test",
                            state="fine",
                            held_out=args.held_out)

    return cbs, optimizer_cls, metrics, trn_data, val_data, tst_data


def model_setup(args):
    experiment_name = args.dataset + '_loss' + args.loss + '_mode' + args.mode + '_eps' + str(args.epoch) + \
                      '_ssllr' + str(args.lr_ssl) + '_clslr' + str(args.lr_cls) + '_bs' + str(args.batch_size) + \
                      '_code' + str(args.code_size) + '_proj' + str(args.proj_size) + \
                      '_label' + str(args.label_eff) + '_sim' + str(args.sim_coeff) + \
                      '_std' + str(args.std_coeff) + '_cov' + str(args.cov_coeff) + \
                      '_temp' + str(args.temp)
    print("Experiment started : \n", experiment_name)
    # log
    if not os.path.isdir(args.logdir):
        os.makedirs(args.logdir)
    log_file_name = os.path.join(args.logdir, args.model_name + f".log")

    # criterion_cls = nn.CrossEntropyLoss()
    # optimizer_cls = tf.keras.optimizers.Adam(args.lr_cls)

    # set up training, validation, and test set
    ssl_data = load_dataset(args.datapath, args.dataset, mode="train", state='ssl',
                            label_efficiency=args.label_eff)
    # tf.config.run_functions_eagerly(True)
    # Loss function
    custom_loss_obj = loss.CustomLoss(temperature=args.temp, sim_coeff=args.sim_coeff,
                                      std_coeff=args.std_coeff, cov_coeff=args.cov_coeff)
    loss_fn = custom_loss_obj.get_loss_fn(args.loss)

    # set up model and optimizers
    ds_info = get_ds_info(args.dataset)
    ssl_model = cm_model(ds_info, code_size=args.code_size, proj_size=args.proj_size,
                         modality_filters=64, l2_rate=1e-4, coverage=0.5)

    trn_data, val_data = load_dataset(args.datapath, args.dataset, mode="train", state="fine",
                                      label_efficiency=args.label_eff)
    tst_data = load_dataset(args.datapath, args.dataset, mode="test", state="fine")

    return ssl_data, trn_data, val_data, tst_data, ssl_model, loss_fn, ds_info, experiment_name
