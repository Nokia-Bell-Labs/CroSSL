import tensorflow as tf

tf.random.set_seed(42)
tf.compat.v1.set_random_seed(42)


def TCN(input_shape, filters, modality_name, code_size, l2_rate):
    initializer = tf.keras.initializers.RandomNormal(seed=0)
    input_x = tf.keras.layers.Input(input_shape)

    x = tf.keras.layers.Conv1D(filters=int(code_size / 4),
                               kernel_size=24,
                               activation="linear",
                               padding="same",
                               # strides=1,
                               kernel_regularizer=tf.keras.regularizers.l2(l2_rate),
                               kernel_initializer=initializer)(input_x)

    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.PReLU(shared_axes=[1])(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Conv1D(filters=int(code_size / 2),
                               kernel_size=16,
                               activation="linear",
                               padding="same",
                               # strides=2,
                               kernel_regularizer=tf.keras.regularizers.l2(l2_rate),
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.PReLU(shared_axes=[1])(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Conv1D(filters=code_size,
                               kernel_size=8,
                               activation="linear",
                               padding="same",
                               # strides=2,
                               kernel_regularizer=tf.keras.regularizers.l2(l2_rate),
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.PReLU(shared_axes=[1])(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    output = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_last', name='global_max_pooling1d')(x)

    return tf.keras.Model(input_x, output, name=modality_name)


def get_modality_encoder(input_shape, modality_name, filters, code_size, l2_rate):
    return TCN(input_shape, filters, modality_name, code_size, l2_rate)
