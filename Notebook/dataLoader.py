import os
import glob
from itertools import combinations, product, islice
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import albumentations as A
import cv2
import numpy as np
import shutil
import albumentations as A

class Loader():
    
    def __init__(self, path, image_shape, batch_size, max_pos_pairs, apply_augmentation=False, augmentation_pourcentage = 1, preprocess_pipeline = None, normalize=False, image_per_celeb=None):
        self.image_per_celeb = image_per_celeb
        self.path = path
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.max_pos_pairs = max_pos_pairs
        self.preprocess_pipeline = preprocess_pipeline
        self.normalize = normalize
        self.apply_augmentation = apply_augmentation
        
        assert 0 < augmentation_pourcentage <= 1 , "pourcentage must be between 0 and 1"
        self.augmentation_pourcentage = augmentation_pourcentage
        self.pos_transform, self.neg_transform = self.createAugmentationModel()

        self.pos_pairs, self.neg_pairs = self.createPairs(path)
        
        print(f"Number of positive pairs: {len(self.pos_pairs)}")
        print(f"Number of negative pairs: {len(self.neg_pairs)}")
        
        if self.apply_augmentation:
            
            augmented_path = path.replace("TensorflowFaces", "TensorflowFaces_augmented")
            shutil.rmtree(augmented_path, ignore_errors=True)
            
            self.pos_pairs_aug, self.neg_pairs_aug = self.createPairs(path, augment = True)
            self.pos_pairs += self.pos_pairs_aug
            self.neg_pairs += self.neg_pairs_aug
            
            print(f"Number of positive pairs after augmentation: {len(self.pos_pairs)}")
            print(f"Number of negative pairs after augmentation: {len(self.neg_pairs)}")
            
        self.dataset = self.createTfDataset(self.pos_pairs, self.neg_pairs, image_shape, batch_size)
        
        selected_pos, total_celeb = self.get_class_celeb_count(self.pos_pairs)
        selected_neg, _ = self.get_class_celeb_count(self.neg_pairs)
        print(f"Used {selected_pos}/{total_celeb} distinct classes for positve pairs")
        print(f"Used {selected_neg}/{total_celeb} distinct classes for negative pairs")
    
    def get_image_per_celeb_count(self):
        
        image_count_map = {}
        
        celeb_list = os.listdir(self.path)
        
        for celeb in celeb_list:
            
            images = glob.glob(os.path.join(self.path, celeb, "*.jpg"))
            count = len(images)
            image_count_map[celeb] = count
    
        return image_count_map
        
    def createAugmentationModel(self):

        pos_transform = A.ReplayCompose([
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=11, border_mode=cv2.BORDER_CONSTANT, p=0.6),
                    A.CropAndPad(percent=(-0.2, 0.2), p=0.6),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                    A.GaussNoise(std_range=(0.02, 0.07), mean_range=(0.0, 0.0), per_channel=True, noise_scale_factor=1.0, p=0.6),
                    A.CoarseDropout(num_holes_range=(1, 2), hole_height_range=(0.10, 0.20), hole_width_range=(0.10, 0.20), fill=0, p=0.5)
                    ])
        
        
        neg_transform = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=7, border_mode=cv2.BORDER_CONSTANT, p=0.3),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.6),
                    A.GaussNoise(std_range=(0.02, 0.04), mean_range=(0.0, 0.0), per_channel=True, noise_scale_factor=1.0, p=0.4)
                    ])
        
        return pos_transform, neg_transform
        
    def augment_pair(self, pairs, mode):
        
        sample = int(self.augmentation_pourcentage * len(pairs))
        pairs = random.sample(pairs, sample)
        
        for i, pair in enumerate(pairs):
            
            modified_pair = []
            
            for j, path in enumerate(pair):
                
                saving_path = path.replace("TensorflowFaces", "TensorflowFaces_augmented")
                os.makedirs(os.path.dirname(saving_path), exist_ok=True)
                
                img = cv2.imread(path)
                img = cv2.resize(img, (128, 128))
                
                if mode == 0:
                    
                    if  j == 0:
                        augmented = self.pos_transform(image=img)
                        img_aug = augmented['image']
                        replay_dict = augmented['replay']

                    else:
                        augmented = self.pos_transform.replay(image=img, saved_augmentations=replay_dict)
                        img_aug = augmented['image']
                        
                else:
                    augmented = self.neg_transform(image=img)
                    img_aug = augmented['image']
                    
                cv2.imwrite(saving_path, img_aug)
                
                modified_pair.append(saving_path)
                
            pairs[i] = tuple(modified_pair)
            
        print("Augmentation done")
        
        return pairs
        
    def createPairs(self, dataset_path, augment=False):

        celeb_folders = os.listdir(dataset_path)

        pos_pairs = []

        for folder in celeb_folders:
            
            images = glob.glob(os.path.join(dataset_path, folder, "*.jpg"))

            if self.image_per_celeb:
                samples = min(len(images), self.image_per_celeb)
                images = random.sample(images, samples)
                
            pos_pairs.extend(combinations(images, 2))

        pos_pairs = random.sample(pos_pairs, min(len(pos_pairs), self.max_pos_pairs))
        
        if augment:
            pos_pairs = self.augment_pair(pos_pairs, mode=0)

        num_pos_pairs = len(pos_pairs)
        print("created pos pair")
        
        i = 0
        neg_pairs=[]
        
        while i < num_pos_pairs:
            celebA = random.choice(celeb_folders)
            celebB = random.choice(celeb_folders)
    
            while celebA == celebB:
                celebB = random.choice(celeb_folders)
    
            celebA_images = glob.glob(os.path.join(dataset_path, celebA, "*.jpg"))
            celebB_images = glob.glob(os.path.join(dataset_path, celebB, "*.jpg"))
    
            neg_pairs.append((random.choice(celebA_images), random.choice(celebB_images)))
            i+=1

        if augment:
            neg_pairs = self.augment_pair(neg_pairs, mode=1)
            
        print("created neg pairs")
        
        return pos_pairs, neg_pairs
            
    def createTfDataset(self, pos_pairs, neg_pairs, image_shape, batch_size):

        x_pos, x_neg = tf.constant(pos_pairs, dtype=tf.string), tf.constant(neg_pairs, dtype=tf.string)

        if self.annotation_type:
            y_pos, y_neg = tf.ones((len(pos_pairs), 1)), tf.zeros((len(neg_pairs), 1))
            
        else:
            y_pos, y_neg = tf.zeros((len(pos_pairs), 1)), tf.ones((len(neg_pairs), 1))

        X = tf.concat([x_pos, x_neg], axis=0)
        Y = tf.concat([y_pos, y_neg], axis=0)

        dataset = tf.data.Dataset.from_tensor_slices((X, Y))

        dataset = dataset.map(lambda paths, label: self.load_and_preprocess_images(paths, label, image_shape),
                          num_parallel_calls=tf.data.AUTOTUNE)
        
        # dataset = dataset.cache()
        
        buffer = min(len(pos_pairs) * 2, 250000)
        dataset = dataset.shuffle(buffer)

        dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset
  
    @tf.function
    def process_file(self, file_path, image_shape):
        
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (image_shape[0], image_shape[1]))
        
        if self.preprocess_pipeline:
            img = self.preprocess_pipeline(img)
            
        else: 
            img = tf.image.rgb_to_grayscale(img) if image_shape[-1] == 1 else img
            img = tf.cast(img, tf.float32)
            img = img / 255.0 if self.normalize else img
            
        return img

    def load_and_preprocess_images(self, image_paths, label, image_shape):
        
        image1 = self.process_file(image_paths[0], image_shape)
        image2 = self.process_file(image_paths[1], image_shape)
        return (image1, image2), label 
    
    def visualize(self, value_range, color_mode_switch):
        
        vmin, vmax = value_range
        
        num_display = 20 
        _, axes = plt.subplots(nrows=num_display, ncols=2, figsize=(15, 5 * num_display))

        for element in self.dataset.take(1):
            (image1, image2), label = element

            image1_np = image1.numpy()
            image2_np = image2.numpy()
            labels_np = label.numpy()

            image1_np = image1_np[:num_display]
            image2_np = image2_np[:num_display]
            labels_np = labels_np[:num_display]

            for i in range(num_display):
                
                img1  = image1_np[i]
                img2 = image2_np[i]
                
                if color_mode_switch:
                    img1 = img1[..., ::-1]
                    img2 = img2[..., ::-1]
                    
                axes[i, 0].imshow(img1, vmin=vmin, vmax=vmax)
                axes[i, 0].axis('off')
                axes[i, 1].imshow(img2, vmin=vmin, vmax=vmax)
                axes[i, 1].axis('off')
                axes[i, 0].set_title(f"Label: {labels_np[i]}")

        plt.tight_layout()
        plt.show()
        
    def get_image_paths_labels(self):

        class_names = sorted(os.listdir(self.path))  # List of subdirectories (player names)
        class_to_label = {name: i for i, name in enumerate(class_names)}  # Assign numeric labels
    
        image_paths = []
        labels = []

        for class_name in class_names:
            class_dir = os.path.join(self.path, class_name)
            for file_name in os.listdir(class_dir):
                if file_name.endswith(('.jpg', '.png', '.jpeg')):
                    image_paths.append(os.path.join(class_dir, file_name))
                    labels.append(class_to_label[class_name])

        return image_paths, labels, class_names  
        
  
    
        image_paths, labels, class_names = self.get_image_paths_labels()

        image_paths = tf.constant(image_paths)
        labels = tf.constant(labels, dtype=tf.int32)

    
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(lambda file_path, label: self.process_file_embedding(file_path, label),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset, class_names
        
    def get_class_celeb_count(self, pairs):
        
        celeb_folders = os.listdir(self.path)
        total_celeb = len(celeb_folders)
        
        visited = set()
        
        for pair in pairs:
            a, b = pair
            class_a, class_b = a.split("/")[4], b.split("/")[4]
            visited.add(class_a) if class_a not in visited else None    
            visited.add(class_b) if class_b not in visited else None
               
        used_celeb = len(visited)
        
        return used_celeb, total_celeb
       
        
       
       
    
        
        
    