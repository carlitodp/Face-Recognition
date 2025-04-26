import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import regularizers

class siameseNetwork(tf.keras.Model):
    
    def __init__(self, base_model, input_size, embedding_dimension=256, apply_augmentation=False, dropout=None, l2_reg=None, name="siamese_network", **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.input_size = input_size
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.embedding_dimension = embedding_dimension

        self.apply_augmentation = apply_augmentation
        
        self.aug_model = self.get_aug_model() if apply_augmentation else None
        self.base_model = self.complete_embedding(base_model)
                
    def call(self, inputs, training=False):
        
        input_1, input_2 = inputs
        
        if self.apply_augmentation and self.aug_model is not None:
            input_1 = self.aug_model(input_1)
            input_2 = self.aug_model(input_2)
        
        emb1 = self.base_model(input_1)
        emb2 = self.base_model(input_2)
        
        return tf.concat([emb1, emb2], axis=1)

    def train_step(self, data):
        
        batch_x, batch_y = data
        
        with tf.GradientTape() as tape:
            batch_y_pred = self(batch_x, training=True)
            loss = self.compiled_loss(batch_y, batch_y_pred)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        self.compiled_metrics.update_state(batch_y, batch_y_pred)
        
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        
        batch_x, batch_y = data

        batch_y_pred = self(batch_x, training=False)
        
        self.compiled_loss(batch_y, batch_y_pred)
        self.compiled_metrics.update_state(batch_y, batch_y_pred)

        return {m.name: m.result() for m in self.metrics}
   
    def get_aug_model(self):
        
        inputs = tf.keras.Input(shape=self.input_size)
    
        x = tf.keras.layers.RandomFlip("horizontal")(inputs)
        x = tf.keras.layers.RandomRotation(0.1, fill_mode='reflect')(x)
        x = tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode='reflect')(x)
        x = tf.keras.layers.RandomZoom(0.1, fill_mode='reflect')(x)
        x = tf.keras.layers.RandomContrast(0.1)(x)
    
        model = tf.keras.Model(inputs=inputs, outputs=x, name="augmentation_model")
        
        return model
            
    def complete_embedding(self, feature_extractor):
        
        x = feature_extractor.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        # x = tf.keras.layers.Dense(64, kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.LeakyReLU()(x)
        # x = tf.keras.layers.Dropout(self.dropout)(x) if self.dropout is not None else x
         
        x = tf.keras.layers.Dense(self.embedding_dimension)(x)
        
        model = Model(inputs = feature_extractor.input, outputs = x, name = "Embedding_model")

        return model
    