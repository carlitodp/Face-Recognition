import tensorflow as tf
from tensorflow.keras import Model
import random
import numpy as np
import os
import cv2

class actionCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, training_data, validation_data, log_dir, batch_size):
        super().__init__()
        self.log_dir = log_dir
        self.activation_path = os.path.join(log_dir, "activation_maps")
        self.batch_size = batch_size
        self.training_data = training_data
        self.validation_data = validation_data
        self.writer = tf.summary.create_file_writer(log_dir)
        
    def on_epoch_end(self, epoch, logs=None):
        model = self.model.base_model
        sample_batch = next(iter(self.training_data.take(1)))
        (img1, img2), label = sample_batch
        random_index = random.randint(0, self.batch_size - 1)
        random_img = img1[random_index]
        img = tf.expand_dims(random_img, axis=0)
        layer_outputs = [layer.output for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        
        for output in layer_outputs:
            layer_name = output.name
            saving_path = os.path.join(self.activation_path, f"epoch_{epoch}", layer_name)
            os.makedirs(saving_path, exist_ok=True)
            
            activation_model = Model(inputs=model.input, outputs=output)
            activations = activation_model.predict(img)
            
            layer_activations = activations[0]
            layer_activations = tf.transpose(layer_activations, perm=[2, 0, 1])
            layer_activations = tf.expand_dims(layer_activations, axis=-1)
            
            num_maps = layer_activations.shape[0]
            if num_maps is None:
                num_maps = int(tf.shape(layer_activations)[0].numpy())
    
            with self.writer.as_default():
                tag = f"{layer_name}/epoch_{epoch}"
                tf.summary.image(tag, layer_activations, step=epoch, max_outputs=num_maps)
                self.writer.flush()
                
            grid = self.create_activation_grid(layer_activations)
            color_grid = cv2.applyColorMap(grid, cv2.COLORMAP_JET)
            cv2.imwrite(f"{saving_path}/activation.jpg", color_grid)
            
    def create_activation_grid(self, activation_maps):
        activation_maps = tf.squeeze(activation_maps, axis=-1).numpy()
        num_maps, H, W = activation_maps.shape
        norm_maps = []
        
        for i in range(num_maps):
            img = activation_maps[i]
            img = img - img.min()
            if img.max() > 0:
                img = img / img.max()
            norm_maps.append(img)
            
        norm_maps = np.array(norm_maps)
        grid_cols = int(np.ceil(np.sqrt(num_maps)))
        grid_rows = int(np.ceil(num_maps / grid_cols))
        grid_img = np.zeros((grid_rows * H, grid_cols * W), dtype=np.float32)
    
        for idx in range(num_maps):
            row = idx // grid_cols
            col = idx % grid_cols
            grid_img[row * H:(row + 1) * H, col * W:(col + 1) * W] = norm_maps[idx]
    
        grid_img = (grid_img * 255).astype(np.uint8)
        return grid_img
