import tensorflow as tf

class contrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, margin=0.7, name="contrastive_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.margin = margin

    def call(self, y_true, y_pred):
       
        positive_loss = (1 - y_true) * tf.square(y_pred)
        base_negative_loss = tf.square(tf.maximum(0.0, self.margin - y_pred))
        negative_weight = tf.where(
            y_pred < self.margin,
            (self.margin - y_pred) / self.margin,
            tf.zeros_like(y_pred)
        )
        negative_loss = y_true * negative_weight * base_negative_loss
        loss = tf.reduce_mean(0.5 * (positive_loss + negative_loss))
        return loss
