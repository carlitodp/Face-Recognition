import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras import Model
from distance import Distance
from accuracy import Accuracy

class siameseNetwork(tf.keras.Model):
    
    def __init__(self, feature_extractor, image_shape, distance_metric, embedding_size):
        super(siameseNetwork, self).__init__()
        self.feature_extractor = feature_extractor
        self.image_shape = image_shape
        self.distance_metric = distance_metric
        self.embedding_size = embedding_size
        assert distance_metric in ["cosine_similarity", "cosine_distance", "euclidean_distance"], "Invalid parameter for distance_metric"
        self.embedding_model = self.createEmbedding(feature_extractor, embedding_size)
        self.siamese_network = self.createSiameseNetwork(self.embedding_model, image_shape)
        
        #Training Metrics
        self.train_dist_sim_metric = Distance(name="train_sim_distance_metric", distance_metric=distance_metric, mode=0)
        self.train_dist_dissim_metric = Distance(name="train_dissim_distance_metric", distance_metric=distance_metric, mode=1)
        self.train_acc = Accuracy(threshold=0.5, name="tran_siamese_acc")
        
        #Validation Metrics
        self.val_dist_sim_metric = Distance(name="val_sim_distance_metric", distance_metric=distance_metric, mode=0)
        self.val_dist_dissim_metric = Distance(name="val_dissim_distance_metric", distance_metric=distance_metric, mode=1)
        self.val_acc = Accuracy(threshold=0.5, name="val_siamese_acc")
        
        #Generic variables
        self.threshold = tf.Variable(0.5, trainable = False, dtype = tf.float32, name = "generic threshold")
    
    def compile(self, optimizer, loss, metrics=None, **kwargs):
        metrics = [self.train_dist_sim_metric, self.train_dist_dissim_metric, self.train_acc, self.val_dist_sim_metric, self.val_dist_dissim_metric, self.val_acc]
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)
    
    def call(self, inputs, training=False):
        return self.siamese_network(inputs, training = training)
        
    def createEmbedding(self, base_model, embedding_size):
    
        l2_reg = regularizers.l2(0.001)
    
        base_model.trainable = True

        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

        x = tf.keras.layers.Dense(1024, kernel_regularizer=l2_reg)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Dropout(0.4)(x)

        x = tf.keras.layers.Dense(embedding_size, kernel_regularizer=l2_reg)(x)
    
        x = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))(x)

        model = Model(inputs = base_model.input, outputs = x, name = "embedding")

        return model
    
    def createSiameseNetwork(self, base_embedding, image_shape):

        first_embedding_input = tf.keras.Input(shape = image_shape, name = "TOWER1")
        second_embedding_input = tf.keras.Input(shape = image_shape, name = "TOWER2")
    
        first_embedding = base_embedding(first_embedding_input)
        second_embedding = base_embedding(second_embedding_input)

        if self.distance_metric == "cosine_distance":
            distance = tf.keras.layers.Lambda(self.cosine_distance)([first_embedding, second_embedding])
        elif self.distance_metric == "cosine_similarity":
            distance = tf.keras.layers.Lambda(self.cosine_similarity)([first_embedding, second_embedding])
        elif self.distance_metric == "euclidean_distance":
            distance = tf.keras.layers.Lambda(self.euclidean_distance)([first_embedding, second_embedding])
        
        
        siamese_network = Model(inputs = [first_embedding_input, second_embedding_input], outputs=distance, name = "siamese_network")

        return siamese_network   
        
    def cosine_distance(self, vectors):
        x, y = vectors
        similarity = tf.reduce_sum(tf.multiply(x, y), axis=-1, keepdims=True)
        similarity = tf.clip_by_value(similarity, -1.0, 1.0)
        distance = (1.0 - similarity) #/ 2.0 #added to bound the distance between 0 and 1
        return distance
    
    def cosine_similarity(self, vectors):
        x, y = vectors
        similarity = tf.reduce_sum(tf.multiply(x, y), axis=-1, keepdims=True)
        similarity = tf.clip_by_value(similarity, -1.0, 1.0)
        return similarity
    
    def euclidean_distance(self, vectors):
        x, y = vectors
        squared_diff = tf.square(x - y)
        sum_squared = tf.reduce_sum(squared_diff, axis=-1, keepdims=True)
        distances = tf.sqrt(sum_squared)
        return distances
    
    def train_step(self, data):
        
        batch_x, batch_y = data
        batch_y = tf.reshape(batch_y, [-1, 1])
        
        with tf.GradientTape() as tape:
            batch_y_pred = self(batch_x, training=True)
            tf.debugging.assert_equal(tf.shape(batch_y), tf.shape(batch_y_pred), message="Shape mismatch")
            loss = self.compiled_loss(batch_y, batch_y_pred)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        self.train_dist_sim_metric.update_state(batch_y, batch_y_pred)
        sim_dist = self.train_dist_sim_metric.result()
        
        self.train_dist_dissim_metric.update_state(batch_y, batch_y_pred)
        dissim_dist = self.train_dist_dissim_metric.result()
        
        self.train_acc.setThreshold(sim_dist)
        self.train_acc.update_state(batch_y, batch_y_pred)
        
        training_acc = self.train_acc.result()
        
        self.threshold.assign(sim_dist)
        
        logs = {
            "threshold": sim_dist,
            "loss": loss,
            "training_loss":loss,
            "training_accuracy": training_acc,
            "training_sim_distance":sim_dist,
            "training_dissim_distance":dissim_dist
        }
        
        return logs
    
    def test_step(self, data, threshold=None):
        
        # print(self.threshold)
        # if threshold is None:
        #     threshold = float(tf.keras.backend.get_value(self.threshold))
        #     print(threshold)
        if threshold is None:
            replica_context = tf.distribute.get_replica_context()
            if replica_context is not None:
                threshold = replica_context.merge_call(
                lambda strategy: strategy.reduce(tf.distribute.ReduceOp.MEAN, self.threshold, axis=None)
                )
            else:
                strategy = tf.distribute.get_strategy()
                threshold = strategy.reduce(tf.distribute.ReduceOp.MEAN, self.threshold, axis=None)
          
        batch_x, batch_y = data
        batch_y = tf.reshape(batch_y, [-1, 1])
        batch_y_pred = self(batch_x, training=False)
        tf.debugging.assert_equal(tf.shape(batch_y), tf.shape(batch_y_pred), message="Shape mismatch")
        loss = self.compiled_loss(batch_y, batch_y_pred)

        self.val_dist_sim_metric.update_state(batch_y, batch_y_pred)
        sim_dist = self.val_dist_sim_metric.result()

        self.val_dist_dissim_metric.update_state(batch_y, batch_y_pred)
        dissim_dist = self.val_dist_dissim_metric.result()

        self.val_acc.setThreshold(threshold)
        self.val_acc.update_state(batch_y, batch_y_pred)
        valid_acc = self.val_acc.result()
        
        logs = {
            "threshold":threshold,
            "loss": loss,
            "accuracy": valid_acc,
            "sim_distance":sim_dist,
            "dissim_distance":dissim_dist
        }
        
        return logs
    