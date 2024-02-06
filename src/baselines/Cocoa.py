import tensorflow as tf

# Modalities
TS_MODALITIES = ["acc", "ACC1", "ACC2", "ACC3", "ACC4", "gyro", "EEG", "EEG1", "EEG2", "EOG", "EMG", "ECG", "EDA",
                 "all", 'HAND_ACC', 'HAND_GYRO', 'ANKLE_ACC', 'ANKLE_GYRO', 'CHEST_ACC', 'CHEST_GYRO', 'heartrate','EEG_Fpz_Cz', 'EEG_Pz_Oz', 'EOG_horizontal', 'EMG_submental']
AUDIO = ["audio", "speech"]
VIDEO = ["video"]


# ------------------- Helper utils ---------------------- #
def get_ts_modality_encoder(input_shape,
                            signal_channel,
                            modality_name,
                            filters,
                            code_size,
                            l2_rate):
    initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
    input = tf.keras.layers.Input(input_shape)

    x = tf.keras.layers.Conv1D(filters=2 * filters,
                               kernel_size=10,
                               activation="linear",
                               padding="same",
                               strides=1,
                               kernel_regularizer=tf.keras.regularizers.l2(l2_rate),
                               kernel_initializer=initializer)(input)

    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.PReLU(shared_axes=[1])(x)

    x = tf.keras.layers.Conv1D(filters=filters,
                               kernel_size=8,
                               activation="linear",
                               padding="same",
                               strides=1,
                               kernel_regularizer=tf.keras.regularizers.l2(l2_rate),
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.PReLU(shared_axes=[1])(x)

    x = tf.keras.layers.Conv1D(filters=code_size,
                               kernel_size=4,
                               activation="linear",
                               padding="same",
                               strides=1,
                               kernel_regularizer=tf.keras.regularizers.l2(l2_rate),
                               kernel_initializer=initializer)(x)
    # output = tf.keras.layers.LayerNormalization()(x)
    output = tf.keras.layers.BatchNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.PReLU(shared_axes=[1])(x)

    return tf.keras.models.Model(input, output, name=modality_name)


def get_audio_encoder(signal_shape,
                      signal_channel,
                      modality_name,
                      filters,
                      code_size,
                      l2_rate):
    return None


def get_video_encoder(signal_length,
                      signal_channel,
                      modality_name,
                      filters,
                      code_size,
                      l2_rate):
    return None


def get_encoder(signal_length,
                signal_channels,
                modality_name,
                code_size,
                l2_rate):
    input = tf.keras.layers.Input((signal_length, signal_channels))

    x = tf.keras.layers.Conv1D(filters=32,
                               kernel_size=10,
                               strides=2,
                               activation="linear",
                               kernel_regularizer=tf.keras.regularizers.l2(l2_rate))(input)
    x = tf.keras.layers.PReLU(shared_axes=[1])(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    x = tf.keras.layers.Conv1D(filters=64,
                               kernel_size=8,
                               strides=2,
                               activation="linear",
                               kernel_regularizer=tf.keras.regularizers.l2(l2_rate))(x)
    x = tf.keras.layers.PReLU(shared_axes=[1])(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    x = tf.keras.layers.Conv1D(filters=128,
                               kernel_size=6,
                               strides=2,
                               activation="linear",
                               kernel_regularizer=tf.keras.regularizers.l2(l2_rate))(x)
    x = tf.keras.layers.PReLU(shared_axes=[1])(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    x = tf.keras.layers.Conv1D(filters=128,
                               kernel_size=6,
                               strides=1,
                               activation="linear",
                               kernel_regularizer=tf.keras.regularizers.l2(l2_rate))(x)
    x = tf.keras.layers.PReLU(shared_axes=[1])(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    x = tf.keras.layers.Conv1D(filters=128,
                               kernel_size=4,
                               strides=1,
                               activation="linear",
                               kernel_regularizer=tf.keras.regularizers.l2(l2_rate))(x)
    x = tf.keras.layers.PReLU(shared_axes=[1])(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    x = tf.keras.layers.Conv1D(filters=code_size,
                               kernel_size=4,
                               strides=1,
                               activation="linear",
                               kernel_regularizer=tf.keras.regularizers.l2(l2_rate))(x)
    x = tf.keras.layers.PReLU(shared_axes=[1])(x)

    output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    ##output = tf.keras.layers.BatchNormalization(epsilon=1e-6)(x)
    return tf.keras.models.Model(input, output, name=modality_name)


# ------------------------------------------------------------------------- #
def create_cocoa_model(win_size, embedding_dim, modality, modality_filters=24, temperature=0.1, mode="ssl",
                                   loss_fn="nce"):
    # modality specific encoders
    mod_encoder = []
    mod_input = []
    for m in modality:
        input_shape = (win_size, 1)
        if m in TS_MODALITIES:
            if m in ["acc", "gyro", "ACC1", "ACC2", "ACC3", "ACC4", 'HAND_ACC', 'HAND_GYRO', 'ANKLE_ACC', 'ANKLE_GYRO',
                     'CHEST_ACC', 'CHEST_GYRO']:
                channels = 3
                input_shape = (win_size, channels)
                encoder = get_ts_modality_encoder(input_shape,
                                                  channels,
                                                  modality_name=m,
                                                  filters=modality_filters,
                                                  code_size=embedding_dim,
                                                  l2_rate=1e-4)
            elif m in ["EEG", "EEG1", "EEG2", "EOG", "EMG", "EDA", "ECG", "heartrate",'EEG_Fpz_Cz', 'EEG_Pz_Oz', 'EOG_horizontal', 'EMG_submental']:
                channels = 1
                input_shape = (win_size, channels)
                """encoder = get_encoder(win_size,
                                      channels,
                                      modality_name=m,
                                      code_size = embedding_dim,
                                      l2_rate=1e-4)"""
                encoder = get_ts_modality_encoder(input_shape,
                                                  channels,
                                                  modality_name=m,
                                                  filters=modality_filters,
                                                  code_size=embedding_dim,
                                                  l2_rate=1e-4)

        elif m == AUDIO:
            encoder = get_audio_encoder(input_shape,
                                        1,
                                        modality_name=m,
                                        filters=modality_filters,
                                        code_size=embedding_dim,
                                        l2_rate=1e-4)

        elif m == VIDEO:
            encoder = get_audio_encoder(input_shape,
                                        1,
                                        modality_name=m,
                                        filters=modality_filters,
                                        code_size=embedding_dim,
                                        l2_rate=1e-4)

        mod_input.append(tf.keras.layers.Input(shape=input_shape))

        x_a = encoder(mod_input[-1])
        x_a = tf.keras.layers.GlobalMaxPooling1D()(x_a)
        x_a = tf.keras.layers.Dense(embedding_dim, activation="linear")(x_a)
        mod_encoder.append(x_a)

    embedding_model = tf.keras.Model(mod_input, mod_encoder)
    if (mode in ["ssl", "fine"]):
        return ContrastiveModel(embedding_model, loss_fn, temperature)
    else:
        return embedding_model


# ------------------------------------------------------------------------- #
class DotProduct(tf.keras.layers.Layer):
    def call(self, x, y):
        x = tf.nn.l2_normalize(x, axis=-1)
        y = tf.nn.l2_normalize(y, axis=-1)
        return tf.linalg.matmul(x, y, transpose_b=True)
        # Changed only because of Barlow Twins test
        # return tf.linalg.matmul(x, y, transpose_a=True)


# ------------------------------------------------------------------------- #
class ContrastiveModel(tf.keras.Model):
    def __init__(self, embedding_model, loss_fn, temperature=1.0, **kwargs):
        super().__init__()
        self.embedding_model = embedding_model
        self._temperature = temperature
        self._similarity_layer = DotProduct()
        self._lossfn = loss_fn

    def train_step(self, data):
        # inp_modality_a, inp_modality_b = data

        with tf.GradientTape() as tape:
            """
            modality_a_embeddings, modality_b_embeddings = self.embedding_model([inp_modality_a, inp_modality_b],
                                                                                training=True)
            similarities = self._similarity_layer(modality_a_embeddings,
                                                  modality_b_embeddings)

            # logits - scale the logits
            #similarities /= self._temperature
            sparse_labels = tf.range(tf.shape(modality_a_embeddings)[0])##barlow
            """
            modality_embeddings = self.embedding_model(data, training=True)
            # Number of Contrasting Modalities
            """num=len(modality_embeddings)
            if num == 2:
                pred = self._similarity_layer(modality_embeddings[0],
                                                      modality_embeddings[1])
            else :
                pred =  modality_embeddings"""
            sparse_labels = tf.range(tf.shape(modality_embeddings[0])[0])
            if self._lossfn in ["cocoa", "mnce2"]:
                pred = modality_embeddings
                pred = tf.nn.l2_normalize(tf.stack(pred), axis=-1)
            elif self._lossfn == "cmc":
                pred = []
                dim_size = len(modality_embeddings)
                for i in range(0, dim_size):
                    for j in range(i, dim_size):
                        pred.append(self._similarity_layer(modality_embeddings[i],
                                                           modality_embeddings[j]))
                        # pred.append(self._similarity_layer(modality_embeddings[j],
                        #                                   modality_embeddings[i]))
                pred = tf.stack(pred)
            else:
                pred = self._similarity_layer(modality_embeddings[0],
                                              modality_embeddings[1])
                if self._lossfn == "barlow":
                    sparse_labels = tf.range(tf.shape(modality_embeddings[0])[1])
            loss = self.compiled_loss(sparse_labels, pred)
            loss += sum(self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # self.compiled_metrics.update_state(sparse_labels, similarities)
        return {m.name: m.result() for m in self.metrics}

    def call(self, input):
        return self.embedding_model(input)


# ------------------------------------------------------------------------- #
def create_AE_network(win_size, embedding_dim, modality, loss_fn, modality_filters=24, combined=0):
    # modality specific encoders
    encoded = []
    mod_input = []
    if combined > 0:
        modality = ["all"]
        channels = combined
    for m in modality:

        if m in TS_MODALITIES:
            if combined == 0:
                if m in ["acc", "gyro", "ACC1", "ACC2", "ACC3", "ACC4"]:
                    channels = 3
                else:
                    channels = 1
            input_shape = (win_size, channels)
            encoder = get_ts_modality_encoder(input_shape,
                                              channels,
                                              modality_name=m,
                                              filters=modality_filters,
                                              code_size=embedding_dim,
                                              l2_rate=1e-4)
        elif m == AUDIO:
            encoder = get_audio_encoder(input_shape,
                                        1,
                                        modality_name=m,
                                        filters=modality_filters,
                                        code_size=embedding_dim,
                                        l2_rate=1e-4)

        elif m == VIDEO:
            encoder = get_audio_encoder(input_shape,
                                        1,
                                        modality_name=m,
                                        filters=modality_filters,
                                        code_size=embedding_dim,
                                        l2_rate=1e-4)

        mod_input.append(tf.keras.layers.Input(shape=input_shape))

        x_a = encoder(mod_input[-1])
        x_a = tf.keras.layers.GlobalMaxPooling1D()(x_a)
        # x_a = tf.keras.layers.Dense(embedding_dim, activation="linear")(x_a)
        encoded.append(x_a)

    # decoded = tf.keras.layers.Concatenate()(encoded)
    decoded = tf.keras.layers.Concatenate()(encoded) if len(modality) > 1 else encoded[0]
    decoded = tf.keras.layers.Dense(len(modality) * channels * embedding_dim, activation='relu')(decoded)
    decoded = tf.keras.layers.Dense(len(modality) * channels * win_size, activation='relu')(decoded)
    decoded = tf.keras.layers.Reshape((len(modality), win_size, channels))(decoded)
    autoencoder = tf.keras.Model(mod_input, decoded, name="autoencoder")
    embedding_model = tf.keras.Model(mod_input, encoded)

    return embedding_model, AutoEncoderModel(autoencoder, loss_fn, combined)


class AutoEncoderModel(tf.keras.Model):
    def __init__(self, autoencoder, loss_fn, combined=False, **kwargs):
        super().__init__()
        self.autoencoder = autoencoder
        self._lossfn = loss_fn
        self.combined = combined

    def train_step(self, data):
        # inp_modality_a, inp_modality_b = data

        with tf.GradientTape() as tape:
            mod_output = self.autoencoder(data, training=True)
            loss = self.compiled_loss(data if self.combined > 0 else tf.stack(data, axis=1), mod_output)
            loss += sum(self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # self.compiled_metrics.update_state(sparse_labels, similarities)
        return {m.name: m.result() for m in self.metrics}

    def call(self, input):
        return self.autoencoder(input)
