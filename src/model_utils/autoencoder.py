import tensorflow as tf
import numpy as np


def build_encoder(data_shape, code_size, name, hidden_layers):
    inputs = tf.keras.Input(data_shape)

    X = inputs
    X = tf.keras.layers.Flatten()(X)
    for hs in hidden_layers:
        X = tf.keras.layers.Dense(hs, activation="relu")(X)
    X = tf.keras.layers.Dense(code_size, activation="sigmoid")(X)
    outputs = X

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=str(name)+"_enc")


def build_decoder(data_shape, code_size, name, hidden_layers):
    inputs = tf.keras.Input((code_size,))

    X = inputs
    for hs in reversed(hidden_layers):
        X = tf.keras.layers.Dense(hs, activation="relu")(X)
    X = tf.keras.layers.Dense(np.prod(data_shape), activation=None)(X)
    outputs = tf.keras.layers.Reshape(data_shape)(X)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=str(name)+"_dec")


# https://gist.github.com/nemanja-rakicevic/551763a6a28149972caccadaafe3bacb
def build_CAE_encoder(data_shape, code_size, name, filters=64, l2_rate=0.0001):
    input_x = tf.keras.layers.Input(data_shape)
    x = tf.keras.layers.Conv1D(filters=2 * filters, kernel_size=3, activation='relu', padding='same')(input_x)
    x = tf.keras.layers.MaxPool1D(4, padding='same')(x)
    x = tf.keras.layers.Conv1D(filters=filters, kernel_size=3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPool1D(4, padding='same')(x)
    x = tf.keras.layers.Flatten()(x)

    x_mean = tf.keras.layers.Dense(code_size*4, activation='tanh')(x)
    embedding = tf.keras.layers.Dense(code_size, activation='linear')(x_mean)
    embedding = tf.keras.layers.LayerNormalization()(embedding)
    return tf.keras.Model(inputs=input_x, outputs=embedding, name=str(name) + "_enc")


def build_CAE_decoder(data_shape, code_size, name, hidden_layers,filters=64):
    input_x = tf.keras.Input((code_size,))
    x = tf.keras.layers.Dense(50)(input_x)
    x = tf.keras.layers.Reshape((25, 2))(x)
    x = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling1D(2)(x)
    x = tf.keras.layers.Conv1D(filters, 1, activation='relu', padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(np.prod(data_shape), activation='linear')(x)
    outputs = tf.keras.layers.Reshape(data_shape)(x)

    return tf.keras.Model(inputs=input_x, outputs=outputs, name=str(name) + "_dec")

def build_TCN_encoder(data_shape, code_size, name, filters=64, l2_rate=0.0001):
    initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
    input_x = tf.keras.layers.Input(data_shape)

    x = tf.keras.layers.Conv1D(filters=2 * filters,
                               kernel_size = 10,
                               activation = "linear",
                               padding = "same",
                               strides = 1,
                               kernel_regularizer = tf.keras.regularizers.l2(l2_rate),
                               kernel_initializer = initializer)(input_x)

    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.PReLU(shared_axes=[1])(x)

    x = tf.keras.layers.Conv1D(filters=filters,
                               kernel_size = 8,
                               activation = "linear",
                               padding = "same",
                               strides = 1,
                               kernel_regularizer = tf.keras.regularizers.l2(l2_rate),
                               kernel_initializer = initializer)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.PReLU(shared_axes=[1])(x)

    x = tf.keras.layers.Conv1D(filters=int(filters/2),
                               kernel_size = 4,
                               activation = "linear",
                               padding = "same",
                               strides = 1,
                               kernel_regularizer = tf.keras.regularizers.l2(l2_rate),
                               kernel_initializer = initializer)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    #x = tf.keras.layers.BatchNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.PReLU(shared_axes=[1])(x)
    x = d(x)
    embedding = tf.keras.layers.Dense(code_size, activation="linear")(x)
    return tf.keras.Model(inputs=input_x, outputs=embedding, name=str(name)+"_enc")


def build_TCN_decoder(data_shape, code_size, name, hidden_layers):
    inputs = tf.keras.Input((code_size,))

    X = inputs
    for hs in reversed(hidden_layers):
        X = tf.keras.layers.Dense(hs, activation="relu")(X)
    X = tf.keras.layers.Dense(np.prod(data_shape), activation=None)(X)
    outputs = tf.keras.layers.Reshape(data_shape)(X)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=str(name)+"_dec")

def create_AE_network(win_size, modality_name, modality_dim, code_size, hidden_layers=[512, 256, 128,64], COMBINED=False, net='FCN'):
    # modality specific encoders
    encoded = []
    mod_input = []
    mod_output = []
    if COMBINED:
        # Single modality
        modality_name = ["all"]
        modality_dim = sum(modality_dim)
    for m in range(len(modality_name)):

        name = modality_name[m]
        channels = modality_dim[m]
        input_shape = (win_size, channels)
        x = tf.keras.layers.Input(shape=input_shape)
        mod_input.append(x)
        if net == 'FCN':
            encoder = build_encoder(input_shape, code_size=code_size, name=name, hidden_layers=hidden_layers)
            decoder = build_decoder(input_shape, code_size=code_size, name=name, hidden_layers=hidden_layers)
        elif net == 'TCN':
            encoder = build_TCN_encoder(data_shape=input_shape, code_size=code_size, name=name)
            decoder = build_decoder(input_shape, code_size=code_size, name=name, hidden_layers=hidden_layers)
        elif net == 'CAE':
            encoder = build_CAE_encoder(data_shape=input_shape, code_size=code_size, name=name)
            decoder = build_CAE_decoder(input_shape, code_size=code_size, name=name, hidden_layers=hidden_layers)
        #encoder.summary()
        #decoder.summary()
        embedding = encoder(mod_input[-1])
        decoded = decoder(embedding)
        mod_output.append(decoded)
        encoded.append(embedding)

    autoencoder = tf.keras.Model(mod_input, mod_output, name="autoencoders")
    encoders = tf.keras.Model(mod_input, encoded)
    return encoders, AutoEncoderModel(autoencoder, COMBINED)


class AutoEncoderModel(tf.keras.Model):
    def __init__(self, autoencoder, COMBINED, **kwargs):
        super().__init__()
        self.autoencoder = autoencoder
        self.COMBINED= COMBINED

    def train_step(self, data):
        # inp_modality_a, inp_modality_b = data

        with tf.GradientTape() as tape:

            mod_output = self.autoencoder(data, training=True)
            loss = self.compiled_loss(tf.stack(data, axis=1) if self.COMBINED else data, mod_output)
            loss += sum(self.losses)

        trainable_vars = self.autoencoder.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # self.compiled_metrics.update_state(sparse_labels, similarities)
        return {m.name: m.result() for m in self.metrics}

    def call(self, input):
        return self.autoencoder(input)
