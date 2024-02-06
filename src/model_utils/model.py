from model_utils.encoders import *
import tensorflow as tf
import numpy as np

tf.random.set_seed(42)
np.random.seed(42)
tf.compat.v1.set_random_seed(42)


def get_encoder(win_size, modality_name, modality_dim, code_size, modality_filters=24, l2_rate=1e-4):
    # modality specific encoders
    mod_encoder = []
    mod_input = []
    i = 0
    for m in modality_name:
        input_shape = (win_size, modality_dim[i])
        encoder = get_modality_encoder(input_shape,
                                       modality_name=m,
                                       filters=modality_filters,
                                       code_size=code_size,
                                       l2_rate=l2_rate)
        # print(encoder.summary())
        mod_input.append(tf.keras.layers.Input(shape=input_shape))

        x_a = encoder(mod_input[-1])
        # x_a = tf.keras.layers.GlobalMaxPooling1D()(x_a)
        # x_a = tf.keras.layers.Dense(code_size, activation="linear", name=m + str(i) + "out")(x_a)
        mod_encoder.append(x_a)
        i += 1
    embedding_model = tf.keras.Model(mod_input, mod_encoder)

    return embedding_model


def get_projector(code_size, proj_size, mod_no):
    input_shape = (mod_no, code_size)
    # TODO
    embeddings = tf.keras.layers.Input(input_shape)
    proj = tf.keras.layers.Flatten()(embeddings)
    proj = tf.keras.layers.Dense(64, activation="relu")(proj)
    proj = tf.keras.layers.Dense(proj_size, activation="relu")(proj)
    # proj = tf.keras.layers.BatchNormalization()(proj)
    proj = tf.keras.layers.Dense(proj_size)(proj)
    # proj = tf.keras.layers.BatchNormalization()(proj)
    proj = tf.keras.layers.LayerNormalization()(proj)
    projection_model = tf.keras.Model(embeddings, proj, name="projection_head")
    # projection_model.summary()
    return projection_model


def get_cm_encoder(encoder, projector, coverage):

    input_x = encoder.input
    x = encoder(input_x)
    idxs = tf.range(len(input_x))
    idx = tf.random.shuffle(idxs)
    x = tf.transpose(tf.gather(x, idx[:coverage]), (1, 0, 2))  # Random modality selection
    x = projector(x)
    return tf.keras.Model(input_x, x)


class DotProduct(tf.keras.layers.Layer):
    def call(self, x, y):
        x = tf.nn.l2_normalize(x, axis=-1)
        y = tf.nn.l2_normalize(y, axis=-1)
        return tf.linalg.matmul(x, y, transpose_b=True)


# ------------------------------------------------------------------------- #
class CMContrastiveModel(tf.keras.Model):
    def __init__(self, embedding_model, projection_model, coverage, masking, masking_threshold=0.9, **kwargs):
        super().__init__()
        self.encoder = embedding_model
        self.projector = projection_model
        self.coverage_rate = coverage
        self._similarity_layer = DotProduct()
        self.masking_strategy = masking
        self.mask_threshold = masking_threshold

    def get_random_indices(self, data_shape):
        # Create random indices from a uniform distribution and then split
        # it into mask and unmask indices.
        if self.masking_strategy == 'random':
            rand = tf.random.uniform(shape=data_shape)
            mask_a = tf.where((rand < self.mask_threshold), 1.0, 0.0)
            rand = tf.random.uniform(shape=data_shape)
            mask_b = tf.where((rand < self.mask_threshold), 1.0, 0.0)

        elif self.masking_strategy == 'spatial':
            # indices = tf.argsort(tf.random.uniform(shape=(data_shape[0], data_shape[1])), axis=1)
            # indices_a = tf.expand_dims(indices[:, :self.coverage_rate], axis=-1)
            # indices_b = tf.expand_dims(indices[:, -self.coverage_rate:], axis=-1)
            indices = tf.random.uniform(shape=(data_shape[0], data_shape[1]))
            indices = tf.where((indices < self.mask_threshold), 1.0, 0.0)
            mask_a = tf.expand_dims(indices, axis=-1)
            mask_a = tf.repeat(mask_a, repeats=data_shape[-1], axis=-1)

            indices = tf.random.uniform(shape=(data_shape[0], data_shape[1]))
            indices = tf.where((indices < self.mask_threshold), 1.0, 0.0)
            mask_b = tf.expand_dims(indices, axis=-1)
            mask_b = tf.repeat(mask_b, repeats=data_shape[-1], axis=-1)

        elif self.masking_strategy == 'temporal':
            # indices = tf.argsort(tf.random.uniform(shape=(data_shape[0], data_shape[1])), axis=1)
            # indices_a = tf.expand_dims(indices[:, :self.coverage_rate], axis=-1)
            # indices_b = tf.expand_dims(indices[:, -self.coverage_rate:], axis=-1)
            indices = tf.random.uniform(shape=(data_shape[0], data_shape[2]))
            indices = tf.where((indices < self.mask_threshold), 1.0, 0.0)
            mask_a = tf.expand_dims(indices, axis=1)
            mask_a = tf.repeat(mask_a, repeats=data_shape[-1], axis=-1)

            indices = tf.random.uniform(shape=(data_shape[0], data_shape[1]))
            indices = tf.where((indices < self.mask_threshold), 1.0, 0.0)
            mask_b = tf.expand_dims(indices, axis=-1)
            mask_b = tf.repeat(mask_b, repeats=data_shape[-1], axis=-1)
        return mask_a, mask_b

    def train_step(self, data):
        with tf.GradientTape() as tape:
            modality_embeddings = self.encoder(data, training=False)

            modality_embeddings = tf.stack(modality_embeddings)
            # dim_size = len(modality_embeddings)
            modality_embeddings = tf.transpose(modality_embeddings, (1, 0, 2))  # Batch x Mod x code_size
            ind_a, ind_b = self.get_random_indices(modality_embeddings.shape)

            # if self.masking_strategy == 'random':
            #    comb_emb_a = modality_embeddings * ind_a
            #    comb_emb_b = modality_embeddings * ind_b
            # elif self.masking_strategy == 'spatial':
            #    comb_emb_a = tf.gather_nd(batch_dims=1, indices=ind_a, params=modality_embeddings)
            #    comb_emb_b = tf.gather_nd(batch_dims=1, indices=ind_b, params=modality_embeddings)

            comb_emb_a = modality_embeddings * ind_a
            comb_emb_b = modality_embeddings * ind_b

            rep_a, rep_b = self.projector(comb_emb_a), self.projector(comb_emb_b)

            loss = self.compiled_loss(rep_a, rep_b)
            loss += sum(self.losses)
            # loss += sum(self.projector.losses)

        trainable_vars = self.encoder.trainable_variables + self.projector.trainable_variables
        gradients = tape.gradient(loss, self.encoder.trainable_variables + self.projector.trainable_variables, )
        self.optimizer.apply_gradients(
            zip(gradients, self.encoder.trainable_variables + self.projector.trainable_variables))
        return {m.name: m.result() for m in self.metrics}


def cm_model(ds_info, code_size, proj_size, modality_filters=32, l2_rate=1e-4, coverage=0.5, masking='random'):
    # mod_coverage = len(ds_info['mod_name']) if masking == 'random' else int(len(ds_info['mod_name']) * coverage)
    mod_coverage = len(ds_info['mod_name'])
    embedding_model = get_encoder(ds_info['win_size'], ds_info['mod_name'], ds_info['mod_dim'],
                                  code_size, modality_filters=modality_filters, l2_rate=l2_rate)
    projection_model = get_projector(code_size, proj_size, mod_coverage)

    return CMContrastiveModel(embedding_model, projection_model, mod_coverage, masking, masking_threshold=coverage)


# ------------------------------------------------------------------------- #
def agg_model(ds_info, encoder_model, code_size, proj_size, coverage=0.5):
    mod_coverage = int(len(ds_info['mod_name']) * coverage)

    projection_model = get_projector(code_size, proj_size, mod_coverage)
    return projection_model, CMContrastiveModel(encoder_model, projection_model, mod_coverage)
