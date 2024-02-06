import tensorflow as tf
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

