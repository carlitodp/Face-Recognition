import tensorflow as tf
import tensorflow_probability as tfp

class Distance(tf.keras.metrics.Metric):
    def __init__(self, distance_fc, mode=0, aggregation_method="mean", name="aggregate_metric", **kwargs):
        super().__init__(name=name, **kwargs)
        self.distance_fc = distance_fc
        self.mode = mode
        self.aggregation_method = aggregation_method
        self.total_value = self.add_weight(name="total_value", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        
        emb1, emb2 = tf.split(y_pred, num_or_size_splits=2, axis=1)

        distance = self.distance_fc(emb1, emb2)
        
        tf.debugging.assert_equal(tf.shape(y_true), tf.shape(distance),
                                  message="y_true and y_pred must have the same shape")
        tf.debugging.assert_equal(tf.rank(y_true), 2,
                                  message="Expected y_true to be of shape (batch_size, 1)")
        tf.debugging.assert_equal(tf.rank(distance), 2,
                                  message="Expected y_pred to be of shape (batch_size, 1)")
        
        if self.mode == 0:
            mask = tf.equal(y_true, 0.0)
        else:
            mask = tf.equal(y_true, 1.0)
        
        y_pred_masked = tf.boolean_mask(distance, mask)
        
        if self.aggregation_method == "mean":
            
            batch_value = tf.cond(tf.size(y_pred_masked) > 0,
                      lambda: tf.reduce_mean(y_pred_masked, axis=None),
                      lambda: tf.constant(0.0, dtype=y_pred_masked.dtype))
            
        elif self.aggregation_method == "median":
            
            batch_value = tfp.stats.percentile(y_pred_masked, 50.0, interpolation="midpoint", axis=None)
            
        elif self.aggregation_method == "quartil":
            if self.mode == 0:
                batch_value = tfp.stats.percentile(y_pred_masked, 75.0, interpolation="midpoint", axis=None)
            else:
                batch_value = tfp.stats.percentile(y_pred_masked, 25.0, interpolation="midpoint", axis=None)
        else:
            raise ValueError("Invalid aggregation_method: " + self.aggregation_method)
        
        self.total_value.assign_add(batch_value)
        self.count.assign_add(1.0)
    
    def result(self):
        return tf.math.divide_no_nan(self.total_value, self.count)
    
    def reset_state(self):
        self.total_value.assign(0.0)
        self.count.assign(0.0)
