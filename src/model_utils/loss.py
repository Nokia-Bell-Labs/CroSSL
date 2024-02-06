import numpy as np
import tensorflow as tf
from typing import Any, Callable, Dict, Optional

class DotProduct(tf.keras.layers.Layer):
    def call(self, x, y):
        x = tf.nn.l2_normalize(x, axis=-1)
        y = tf.nn.l2_normalize(y, axis=-1)
        return tf.linalg.matmul(x, y, transpose_b=True)


class CustomLoss:

    def __init__(
            self,
            temperature=0.5,
            tau=0.01,
            beta=2,
            sim_coeff=25,
            std_coeff=25,
            cov_coeff=1,
            std_const=1e-4,
            lambd=3.9e-3,
            scale_loss=1 / 32,
            reduction=tf.keras.losses.Reduction.NONE
    ):
        self.temperature = temperature,
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.beta = beta
        self.tau = tau
        self.lambd = lambd
        self.scale_loss = scale_loss
        # Please double-check `reduction` parameter
        self.criterion = tf.keras.losses.BinaryCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.SUM)
        self.std_const = std_const
        self.reduction = reduction
        self._similarity_fn = DotProduct()
        self.criterion = tf.keras.losses.BinaryCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.SUM)

    def tf_cov(self, x):
        # print("tf_cov >>> x shape ", x.shape)
        mx = (x - tf.math.reduce_mean(x, axis=0, keepdims=True))
        cov_x = tf.matmul(tf.transpose(mx), mx) / tf.cast(x.shape[-1], tf.float32)
        # print("tf_cov >>> cov_x shape ", cov_x.shape)
        return cov_x

    def get_config(self) -> Dict[str, Any]:
        config = {
            "std_const": self.std_const,
            "lambda_": self.lambda_,
            "mu": self.mu,
            "nu": self.nu,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def off_diagonal(self, x):
        n = tf.shape(x)[0]
        flattened = tf.reshape(x, [-1])[:-1]
        off_diagonals = tf.reshape(flattened, (n - 1, n + 1))[:, 1:]
        off_diag = tf.reshape(off_diagonals, [-1])
        return off_diag

    def cov_loss_each(self, z, batch_size):
        # cross-correlation matrix axa
        c = tf.matmul(z, z, transpose_a=True)
        c = c / tf.cast(batch_size - 1, dtype="float32")

        num_features = tf.shape(c)[0]

        off_diag_c = self.off_diagonal(c)
        off_diag_c = tf.math.pow(off_diag_c, 2)

        off_diag_c = tf.math.reduce_sum(off_diag_c) / tf.cast(
            num_features, tf.float32
        )

        return off_diag_c

    def mean_center_columns(self, x):
        col_mean = tf.math.reduce_mean(x, axis=0)

        norm_col = x - col_mean
        return norm_col

    def get_loss_fn(self, loss_type):
        loss = None
        if loss_type == "nce":
            def loss(r1, r2):
                dot_prod = self._similarity_fn(r1, r2)
                all_sim = tf.math.exp(dot_prod / self.temperature)
                logits = tf.divide(
                    tf.linalg.tensor_diag_part(all_sim), tf.reduce_sum(all_sim, axis=1))
                #print(logits)
                lbl = np.ones(dot_prod.shape[0])
                error = self.criterion(y_pred=logits, y_true=lbl)
                return error

            # Debiased Contrastive Learning
        elif loss_type in ["dcl", "harddcl"]:
            def loss(r1, r2):
                # dcl: from Debiased Contrastive Learning paper: https://github.com/chingyaoc/DCL/
                # harddcl: from ICLR2021 paper: Contrastive LEarning with Hard Negative Samples
                # https://www.groundai.com/project/contrastive-learning-with-hard-negative-samples
                # reweight = (beta * neg) / neg.mean()
                # Neg = max((-N * tau_plus * pos + reweight * neg).sum() / (1 - tau_plus), e ** (-1 / t))
                # hard_loss = -log(pos.sum() / (pos.sum() + Neg))
                sim_mat = self._similarity_fn(r1,r2)
                N = sim_mat.shape[0]
                all_sim = tf.math.exp(sim_mat / self.temperature)
                pos_sim = tf.linalg.tensor_diag_part(all_sim)

                tri_mask = np.ones(N ** 2, dtype=np.bool).reshape(N, N)
                tri_mask[np.diag_indices(N)] = False
                neg_sim = tf.reshape(tf.boolean_mask(all_sim, tri_mask), [N, N - 1])

                reweight = 1.0
                if loss_type == "harddcl":
                    reweight = (self.beta * neg_sim) / tf.reshape(tf.reduce_mean(neg_sim, axis=1), [-1, 1])
                    if self.beta == 0:
                        reweight = 1.0

                Ng = tf.divide(
                    tf.multiply(self.tau* (1 - N), pos_sim) + tf.reduce_sum(tf.multiply(reweight, neg_sim), axis=-1),
                    (1 - self.tau))
                #print(Ng)
                # constrain (optional)
                Ng = tf.clip_by_value(Ng, clip_value_min=(N - 1) * np.e ** (-1 / self.temperature[0]),
                                      clip_value_max=tf.float32.max)
                error = tf.reduce_mean(- tf.math.log(pos_sim / (pos_sim + Ng)))
                return error
                # Contrasting More than two dimenstions
        elif loss_type == "cocoa":
            def loss(ytrue, ypred):
                batch_size, dim_size = ypred.shape[1], ypred.shape[0]
                # Positive Pairs
                pos_error = []
                for i in range(batch_size):
                    sim = tf.linalg.matmul(ypred[:, i, :], ypred[:, i, :], transpose_b=True)
                    sim = tf.subtract(tf.ones([dim_size, dim_size], dtype=tf.dtypes.float32), sim)
                    sim = tf.exp(sim / self.temperature)
                    pos_error.append(tf.reduce_mean(sim))
                # Negative pairs
                neg_error = 0
                for i in range(dim_size):
                    sim = tf.cast(tf.linalg.matmul(ypred[i], ypred[i], transpose_b=True), dtype=tf.dtypes.float32)
                    sim = tf.exp(sim / self.temperature)
                    # sim = tf.add(sim, tf.ones([batch_size, batch_size]))
                    tri_mask = np.ones(batch_size ** 2, dtype=np.bool).reshape(batch_size, batch_size)
                    tri_mask[np.diag_indices(batch_size)] = False
                    off_diag_sim = tf.reshape(tf.boolean_mask(sim, tri_mask), [batch_size, batch_size - 1])
                    neg_error += (tf.reduce_mean(off_diag_sim, axis=-1))

                # error = (pos_error + neg_error)/(batch_size)
                error = tf.multiply(tf.reduce_sum(pos_error), self.scale_loss) + self.lambd * tf.reduce_sum(
                    neg_error)
                return error

        elif loss_type == 'vicreg':
            def loss(za, zb):
                # compute the diagonal
                batch_size = tf.shape(za)[0]
                # distance loss to measure similarity between representations
                sim_loss = tf.keras.losses.MeanSquaredError(reduction="none")(za, zb)

                za = self.mean_center_columns(za)
                zb = self.mean_center_columns(zb)

                # std loss to maximize variance(information)
                std_za = tf.sqrt(tf.math.reduce_variance(za, 0) + self.std_const)
                std_zb = tf.sqrt(tf.math.reduce_variance(zb, 0) + self.std_const)

                std_loss_za = tf.reduce_mean(tf.math.maximum(0.0, 1 - std_za))
                std_loss_zb = tf.reduce_mean(tf.math.maximum(0.0, 1 - std_zb))

                std_loss = std_loss_za / 2 + std_loss_zb / 2

                off_diag_ca = self.cov_loss_each(za, batch_size)
                off_diag_cb = self.cov_loss_each(zb, batch_size)

                # covariance loss(1d tensor) for redundancy reduction
                cov_loss = off_diag_ca + off_diag_cb

                error_value = (
                        self.sim_coeff * sim_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss
                )

                return error_value

        elif loss_type == "mse":
            def loss(ytrue, ypred):
                reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(ytrue, ypred)))
                return reconstruction_error
        else:
            raise ValueError("Undefined loss function.")

        return loss
