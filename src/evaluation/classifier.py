import tensorflow as tf
tf.random.set_seed(42)
tf.compat.v1.set_random_seed(42)

def get_classifier(encoder, class_size, encoder_only= False, projector=None, coverage=0, dropout=0.1):
    if encoder_only:
        input_x = encoder.input
        x = encoder(input_x)
        x = tf.keras.layers.Concatenate()(x)
    else:
        input_x = encoder.input
        x = encoder(input_x, training=False)
        x = tf.stack(x)
        x = tf.transpose(x, (1, 0, 2))
        #idxs = tf.range(len(input_x))
        #idx = tf.random.shuffle(idxs)
        #x = tf.transpose(tf.gather(x, [0,2,4]), (1, 0, 2))  # Random modality selection
        x = projector(x, training=False)


    #x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)  # Regularize with dropout
    #x = tf.keras.layers.Dense(32, activation="relu")(x)
    #x = tf.keras.layers.Dropout(dropout)(x)  # Regularize with dropout
    classifier_model = tf.keras.layers.Dense(class_size, activation="softmax", name="classifier_last_dense")(x)

    # Combine encoder and extra layers
    c_model = tf.keras.Model(input_x, classifier_model)

    return c_model


