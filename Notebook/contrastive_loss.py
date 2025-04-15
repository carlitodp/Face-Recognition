import tensorflow as tf

class contrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, distance_fc, margin=0.7, hard_mining=False, name="contrastive_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.distance_fc = distance_fc
        self.margin = margin
        self.hard_mining = hard_mining

    def call(self, y_true, y_pred):
        emb1, emb2 = tf.split(y_pred, num_or_size_splits=2, axis=1)
        distance = self.distance_fc(emb1, emb2)

        positive_loss = (1 - y_true) * tf.square(distance)
        negative_loss = y_true * tf.square(tf.maximum(0.0, self.margin - distance))

        if self.hard_mining:
            # Keep only positives with distance >= margin
            # and negatives with distance <= margin
            pos_mask = tf.logical_and(tf.equal(y_true, 0.0), distance >= self.margin)
            neg_mask = tf.logical_and(tf.equal(y_true, 1.0), distance <= self.margin)
            keep_mask = tf.logical_or(pos_mask, neg_mask)

            positive_loss = tf.where(keep_mask, positive_loss, 0.0)
            negative_loss = tf.where(keep_mask, negative_loss, 0.0)

        loss = tf.reduce_mean(0.5 * (positive_loss + negative_loss))
        return loss
