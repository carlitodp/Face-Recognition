import tensorflow as tf
 
class circleLoss(tf.keras.losses.Loss):
    def __init__(self, margin, gamma, name="circle_loss", reduction=tf.keras.losses.Reduction.AUTO):
        super().__init__(name=name, reduction=reduction)
        self.margin = margin
        self.name = name
        self.gamma = gamma
       
    def call(self, y_true, y_pred):
       
        y_pred_true = tf.boolean_mask(y_pred, y_true == 1.0)
        y_pred_false = tf.boolean_mask(y_pred, y_true == 0.0)
 
        alpha_p = tf.maximum(1 + self.margin - y_pred_true, 0)
        delta_p = 1 - self.margin
       
        alpha_n = tf.maximum(y_pred_false + self.margin, 0)
        delta_n = self.margin
       
        pos_similarity = tf.math.exp(self.gamma * (alpha_p * y_pred_true - delta_p))
        pos_similarity = tf.reduce_sum(pos_similarity)
       
        neg_similarity = tf.math.exp(self.gamma * (alpha_n * y_pred_false - delta_n))
        neg_similarity = tf.reduce_sum(neg_similarity)
       
        loss = tf.math.log(1 + pos_similarity * neg_similarity)
       
        return loss