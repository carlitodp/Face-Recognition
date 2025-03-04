import tensorflow as tf

class Distance(tf.keras.metrics.Metric):
    def __init__(self, mode, distance_metric, name="distance_metric", **kwargs): #mode = 0 for sim or 1 for dissim
        super().__init__(name=name, **kwargs)
        self.mode = mode
        self.distance_metric = distance_metric
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        
        if self.distance_metric in ["cosine_distance", "euclidean_distance"]:
            mask = tf.cond(tf.equal(self.mode, 0),
                            lambda: tf.equal(y_true, 0.0),
                            lambda: tf.equal(y_true, 1.0))
            
        if self.distance_metric == "cosine_similarity":
            mask = tf.cond(tf.equal(self.mode, 0),
                            lambda: tf.equal(y_true, 1.0),
                            lambda: tf.equal(y_true, 0.0))

        
        distances = tf.boolean_mask(y_pred, mask)
    
        self.total.assign_add(tf.reduce_sum(distances))
        self.count.assign_add(tf.cast(tf.size(distances), tf.float32))
    
    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)
    
    def result(self):
        result = tf.math.divide_no_nan(self.total, self.count)
        return result
