import tensorflow as tf

class Accuracy(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name="accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.total = self.add_weight(name="total", initializer="zeros", dtype=tf.float32)
        self.correct = self.add_weight(name="correct", initializer="zeros", dtype=tf.float32)

    def setThreshold(self, threshold):

        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_pred_binary = tf.cast(y_pred >= self.threshold, tf.float32)

        correct_predictions = tf.reduce_sum(tf.cast(y_pred_binary == y_true, tf.float32))
        total_samples = tf.cast(tf.size(y_true), tf.float32)

        self.correct.assign_add(correct_predictions)
        self.total.assign_add(total_samples)

    def result(self):
        return self.correct / (self.total + tf.keras.backend.epsilon())

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)
