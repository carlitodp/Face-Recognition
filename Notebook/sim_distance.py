import tensorflow as tf

class similarityDistances():
    def __init__(self, normalize=False):
        self.normalize = normalize
    
    def cosine_distance(self, x, y):
        
        if self.normalize:
            x = tf.nn.l2_normalize(x, axis=1)
            y = tf.nn.l2_normalize(y, axis=1)
            
        similarity = tf.reduce_sum(tf.multiply(x, y), axis=1, keepdims=True)
        similarity = tf.clip_by_value(similarity, -1.0, 1.0)
        distance = (1.0 - similarity)
        return distance
    
    def cosine_similarity(self, x, y):
        
        if self.normalize:
            x = tf.nn.l2_normalize(x, axis=1)
            y = tf.nn.l2_normalize(y, axis=1)
            
        similarity = tf.reduce_sum(tf.multiply(x, y), axis=1, keepdims=True)
        similarity = tf.clip_by_value(similarity, -1.0, 1.0)
        return similarity
    
    def euclidean_distance(self, x, y):
        
        if self.normalize:
            x = tf.nn.l2_normalize(x, axis=1)
            y = tf.nn.l2_normalize(y, axis=1)
            
        squared_diff = tf.square(x - y)
        sum_squared = tf.reduce_sum(squared_diff, axis=1, keepdims=True)
        distances = tf.sqrt(sum_squared)
        return distances