import tensorflow as tf
import os
from tensorboard.plugins import projector
import numpy as np

class callBacks(tf.keras.callbacks.Callback):
    
    def __init__(self, log_dir, embedding_dataset, class_names):
        super(callBacks, self).__init__()
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self.embedding_dataset = embedding_dataset
        self.class_names = class_names
    
    
    def on_epoch_end(self, epoch, logs=None):
        
        logs = logs or {}
    
        training_threshold = logs.get("threshold", 0)
        training_loss = logs.get("training_loss", 0)
        training_accuracy = logs.get("training_accuracy", 0)
    
        train_sim_distance = logs.get("training_sim_distance", 0)
        train_dissim_distance = logs.get("training_dissim_distance", 0)
    
        validation_threshold = logs.get("val_threshold", 0)
        validation_loss = logs.get("val_loss", 0)
        validation_accuracy = logs.get("val_accuracy", 0)
    
        valid_sim_distance = logs.get("val_sim_distance", 0)
        valid_dissim_distance = logs.get("val_dissim_distance", 0)
    
        with self.writer.as_default():
            tf.summary.scalar('train_loss', training_loss, step=epoch)
            tf.summary.scalar('train_acc', training_accuracy, step=epoch)
            tf.summary.scalar('valid_loss', validation_loss, step=epoch)
            tf.summary.scalar('valid_accuracy', validation_accuracy, step=epoch)

        with self.writer.as_default():
                        tf.summary.text("Training Epoch Sim Logs ",
                        f"Epoch: {epoch} Sim: {train_sim_distance:.3f}  Dissim: {train_dissim_distance:.3f} threshold: {training_threshold:.3f}",
                        step=epoch)

        with self.writer.as_default():
                        tf.summary.text("Validation Epoch Sim Logs ",
                        f"Epoch: {epoch} Sim: {valid_sim_distance:.3f}  Dissim: {valid_dissim_distance:.3f} threshold: {validation_threshold:.3f}",
                        step=epoch)
                        

        if epoch % 5 == 0:
            siamese_path_model = os.path.join(self.log_dir, f"siamese_model_epoch_{epoch}.h5")
            self.model.save_weights(siamese_path_model)

            embedding_path_model = os.path.join(self.log_dir, f"embedding_model_epoch_{epoch}.h5")
            self.model.embedding_model.save_weights(embedding_path_model)

        self.save_embeddings(epoch, self.class_names)
             
    def projectEmbeddings(self):

        embeddings_list = []
        labels_list = []

        for images, labels in self.embedding_dataset:
            embeddings = self.model.embedding_model(images) 
            embeddings_list.append(embeddings.numpy()) 
            labels_list.extend(labels.numpy())

        embeddings_array = np.concatenate(embeddings_list, axis=0)
        labels_array = np.array(labels_list)

        return embeddings_array, labels_array

    def save_embeddings(self, epoch, class_names):

        embeddings, labels = self.projectEmbeddings()

        embedding_var = tf.Variable(embeddings, name=f"embedding_epoch_{epoch}")

        checkpoint = tf.train.Checkpoint(embedding=embedding_var)
        checkpoint.save(os.path.join(self.log_dir, f"embedding_epoch_{epoch}.ckpt"))

        metadata_path = os.path.join(self.log_dir, f"metadata_epoch_{epoch}.tsv")
        
        with open(metadata_path, "w") as f:
            for label in labels:
                f.write(f"{class_names[label]}\n")  

        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        embedding.metadata_path = f"metadata_epoch_{epoch}.tsv"

        projector.visualize_embeddings(self.log_dir, config)
        
