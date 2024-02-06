#import tensorflow
#import printer as printer
#from sup_data_helper import *
from sklearn.model_selection import train_test_split
import argparse
import os
import tensorflow as tf
import pandas as pd

from baselines.sup_data_helper import load_har_ds_sup, load_sleep4_ds_sup


# author: Alexander Hoelzemann - alexander.hoelzemann@uni-siegen.de


###############################################################
##            Class for customized callbacks                 ##
##          Can be called each epoch, batch etc.             ##
###############################################################

class CustomCallbacks(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass


class CustomMetrics():
    # copied from unnir's post https://github.com/keras-team/keras/issues/5400

    def recall_m(y_true, y_pred):
        true_positives = tf.keras.backend.sum(
            tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = tf.keras.backend.sum(
            tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        true_positives = tf.keras.backend.sum(
            tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        predicted_positives = tf.keras.backend.sum(
            tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        return precision

    def f1_m(y_true, y_pred):
        precision = CustomMetrics.precision_m(y_true, y_pred)
        recall = CustomMetrics.recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))



class DeepConvLSTM:

    ###############################################################
    ##                    Class-Initialization                   ##
    ##                   and Network definition                  ##
    ###############################################################

    def __init__(self, Xs, Ys, X_test=None, Y_test=None, X_val=None, y_val=None, modelname='DeepConvLSTM', verbose=None, epochs=None,
                 batch_size=None, validation_split=None, learning_rate=0.001, regularization_rate=0.01,
                 regularization=tf.keras.regularizers.l2,
                 optimizer=tf.keras.optimizers.Adam, use_batch_norm=False, rnn_size=128,
                 num_rnn_layers=2, dropout_rate=0.1, filter_size=[64, 128, 256], samples_per_convolution_step=5,
                 samples_per_pooling_step=0, weight_init='lecun_uniform'):

        '''
        :param Xs: x_train: ndarray
        :param Ys: y_train: ndarray
        :param X_test: x_test: ndarray
        :param Y_test: y:test: ndarray
        :param modelname: string
        :param verbose: int between 0 and 2
        :param epochs: number of training epochs: int
        :param batch_size: int
        :param validation_split: variable that defines the percentage of training data used for validation: float between 0.01 and 0.99
        :param window_length: window length as number of samples: int
        :param learning_rate: float
        :param regularization: desired regularization as object from tensorflow.keras.regularizers
        :param random_seed: int
        :param optimizer: desired optimizers as object from tensorflow.keras.regularizers
        :param use_batch_norm: set True if batch normalization should be used : boolean
        :param rnn_size: int
        :param num_rnn_layers: int
        :param dropout_rate: float between 0.1 and 0.9
        :param filter_size: list of filter sizes for every convolutational layer - every element is an int
        :param samples_per_convolution_step: int
        :param samples_per_pooling_step: int - if you would like to use MaxPooling, change to an int > 0
        :param weight_init: string
        '''

        self.X_train = Xs
        self.y_train = Ys
        self.X_validation = X_val
        self.y_validation = y_val
        self.X_test = X_test
        self.y_test = Y_test
        print("X_train {}, y_train {}, X_test {}, y_test {}".format(Xs.shape, Ys.shape, X_test.shape, Y_test.shape))
        if validation_split > 0.0 and X_val is None:
            self.X_train, self.X_validation, self.y_train, self.y_validation = train_test_split(self.X_train,
                                                                                                self.y_train,
                                                                                                test_size=validation_split,
                                                                                                shuffle=False)
        else:
            pass

        self.verbose = verbose
        self.epochs = epochs
        self.validation_split = validation_split

        # Settings for HyperParameters
        self.batch_depth = self.X_train.shape[0]
        self.batch_length = self.X_train.shape[1]
        self.n_channels = self.X_train.shape[2]
        self.n_classes = self.y_train.shape[1]
        self.batch_size = batch_size
        self.weight_init = weight_init
        self.optimizer = optimizer(learning_rate=learning_rate)
        self.regularization = regularization(l=regularization_rate)
        self.rnn_size = rnn_size
        self.num_rnn_layers = num_rnn_layers
        self.dropout_rate = dropout_rate
        self.filter_size = filter_size
        self.num_cnn_layers = len(self.filter_size)
        self.modelname = modelname
        self.use_batch_norm = use_batch_norm
        self.kernel_size = (samples_per_convolution_step, self.n_channels)
        self.pool_size = (samples_per_pooling_step, 1)

    def init_network(self):

        inputs = tf.keras.Input(shape=(self.batch_length, self.n_channels))
        if self.use_batch_norm:
            x = tf.keras.layers.BatchNormalization(name='input/batch_norm',
                                                           input_shape=(self.batch_length, self.n_channels))(inputs)
        x = tf.keras.layers.Reshape(name='reshape_to_3d', target_shape=(self.batch_length, self.n_channels, 1))(
            inputs)
        for cnn_layer in range(0, self.num_cnn_layers):
            x = tf.keras.layers.Convolution2D(name='conv2d_' + str(cnn_layer),
                                                      filters=self.filter_size[cnn_layer], kernel_size=self.kernel_size,
                                                      padding='same',
                                                      kernel_regularizer=self.regularization,
                                                      bias_regularizer=self.regularization,
                                                      kernel_initializer=self.weight_init)(x)

            if self.use_batch_norm:
                x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
        if self.pool_size[0] > 0:
            x = tf.keras.layers.MaxPooling2D(self.pool_size[0])(x)
        x = tf.keras.layers.Reshape(name="reshape_to_1d",
                                            target_shape=(x.shape[1], self.filter_size[-1] * self.n_channels))(x)
        for rnn_layer in range(0, self.num_rnn_layers):
            if rnn_layer == 0:
                x = tf.keras.layers.LSTM(self.rnn_size,
                                                 return_sequences=True)(x)
            else:
                x = tf.keras.layers.LSTM(self.rnn_size, return_sequences=False)(x)

        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        outputs = tf.keras.layers.Dense(self.n_classes, activation='softmax')(x)
        self.neural_network = tf.keras.Model(inputs, outputs, name=self.modelname)
        self.neural_network.compile(loss='categorical_crossentropy', optimizer=self.optimizer,
                                    metrics=['accuracy', CustomMetrics.f1_m, CustomMetrics.precision_m,
                                             CustomMetrics.recall_m])

        #printer.write('New network intialized', 'blue')

    def fit(self):
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=2, restore_best_weights=True)
        if self.validation_split > 0.0:
            es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=2, restore_best_weights=True)
            return self.neural_network.fit(self.X_train, self.y_train, epochs=self.epochs,
                                           batch_size=self.batch_size,
                                           verbose=self.verbose,
                                           validation_data=(self.X_validation, self.y_validation),
                                           callbacks=[CustomCallbacks(),es])
        else:
            es = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5, verbose=2, restore_best_weights=True)
            return self.neural_network.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size,
                                           verbose=self.verbose, callbacks=[CustomCallbacks(),es])

    ###############################################################
    ##            Methods to evaluate the model                  ##
    ###############################################################

    def evaluate_model(self):
        self.evaluation_results = {}
        self.evaluation_results['loss'], self.evaluation_results['accuracy'], \
        self.evaluation_results['f1_score'] = self.neural_network.evaluate(self.X_train, self.y_train,
                                                                           verbose=self.verbose)

    ###############################################################
    ##      execute the experiment: and train the neural network ##
    ###############################################################

    def execute_experiment(self):

        self.init_network()
        history = self.fit()
        predictions = self.neural_network.predict(x=self.X_test)
        result = self.neural_network.evaluate(self.X_test, self.y_test, verbose=self.verbose)
        return history, predictions, result


# Example
# my_neural_network = DeepConvLSTM(Xs=X_train, Ys=y_train, X_test=X_test, Y_test=y_test,
#                                 verbose=2, epochs=30, batch_size=64, window_length=50, validation_split=0.1)
# history, predictions = my_neural_network.execute_experiment()
#
# Compare predictions to self.y_test

parser = argparse.ArgumentParser(description='interface of running experiments for DEEPCONVLSTM')
parser.add_argument('--datapath', type=str, default='../../data', help='prefix path to data directory')
parser.add_argument('--output', type=str, default='../../output', help='prefix path to output directory')
parser.add_argument('--dataset', type=str, default='USCHAR', help='dataset name ')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--gpu_device', type=str, default='1', help='GPU')
parser.add_argument('--batch_size', type=int, default=64, help='batch size of training')
parser.add_argument('--epoch', type=int, default=50, help='number of training epochs')
def setup_system(gpu, output_path):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
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

    working_directory = os.path.join(output_path)
    os.makedirs(working_directory, exist_ok=True)
    os.makedirs(os.path.join(working_directory, 'models/'), exist_ok=True)
    os.makedirs(os.path.join(working_directory, 'logs/'), exist_ok=True)
    os.makedirs(os.path.join(working_directory, 'results/'), exist_ok=True)
    os.makedirs(os.path.join(working_directory, 'plots/'), exist_ok=True)
    return working_directory

if __name__ == "__main__":
    args = parser.parse_args()

    working_dir = setup_system(args.gpu_device, args.output)
    # read data
    if args.dataset == 'USCHAR':
        X_train, y_train = load_har_ds_sup(os.path.join(args.datapath,"har"), mode='train')
        X_test, y_test = load_har_ds_sup(os.path.join(args.datapath, "har"), mode='test')
    elif args.dataset == "SLEEPEDF":
        X_train, lbl = load_sleep4_ds_sup(os.path.join(args.datapath, "sleep"), mode='train')
        X_test, y_test = load_sleep4_ds_sup(os.path.join(args.datapath, "sleep"), mode='test')
    elif args.dataset in ["pamap2"]:
        X_train, y_train = load_pamap2_ds_sup(os.path.join(args.datapath, "pamap2"), mode='train')
        X_val, y_val = load_pamap2_ds_sup(os.path.join(args.datapath, "pamap2"), mode='val')
        X_test, y_test = load_pamap2_ds_sup(os.path.join(args.datapath, "pamap2"), mode='test')

    elif args.dataset in ["WESAD"]:
        X_train, y_train = load_wesad_ds_sup(os.path.join(args.datapath, "wesad"), mode='train')
        X_test, y_test = load_wesad_ds_sup(os.path.join(args.datapath, "wesad"), mode='test')
    elif args.dataset in ["OPP"]:
        X_train, y_train = load_opportunity_ds_sup(os.path.join(args.datapath, "opportunity"), mode='train')
        X_test, y_test = load_opportunity_ds_sup(os.path.join(args.datapath, "opportunity"), mode='test')
    # ccreate the model
    my_neural_network = DeepConvLSTM(Xs=X_train, Ys=y_train, X_test=X_test, Y_test=y_test, X_val=X_val, y_val= y_val, learning_rate=args.lr,
                                     verbose=1, epochs=args.epoch, batch_size=args.batch_size, validation_split=0.1)
    history, predictions,result= my_neural_network.execute_experiment()

    #print(cm.f1_m(y_test, np.transpose(predictions)[0]))
    print(result)
    print("TEST evaluation : LOSS : {} -  ACC : {} - F1-score : {} - Precision : {}, Recall : {}".format(result[0], result[1], result[2], result[3], result[4]))
    # execute experiments
    # save result
    df_columns = [ 'DATASET', 'BATCH', 'EPOCH', 'LOSS', 'ACC', 'FSCORE']
    report = [[args.dataset, args.batch_size, args.epoch, result[0], result[1], result[2]]]

    df = pd.DataFrame(report, columns=df_columns)
    df.to_csv(os.path.join(working_dir, "results", "report_deep.csv"), mode='a', header=False)
