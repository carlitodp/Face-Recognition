import os
import glob
from itertools import combinations, product, islice
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import albumentations as A
import cv2
import numpy as np

class Loader():
    
    def __init__(self, path, image_shape, batch_size, max_pos_pairs, apply_augmentation=False, annotation_type = None, preprocess_pipeline = None, normalize=False):
        self.path = path
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.max_pos_pairs = max_pos_pairs
        self.annotation_type = annotation_type
        self.preprocess_pipeline = preprocess_pipeline
        self.normalize = normalize
        assert annotation_type in [None, "angular"], "annotation_type must be None or 'angular' "
        self.apply_augmentation = apply_augmentation
        self.transform = self.createAugmentationModel()

        pos_pairs, neg_pairs = self.createPairs(path)
        self.dataset = self.createTfDataset(pos_pairs, neg_pairs, image_shape, batch_size)
        
    def createAugmentationModel(self):

        transform = A.Compose([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),

            A.OneOf([
                A.Blur(blur_limit=(5, 7), p=1.0),
                A.SaltAndPepper(amount=(0.2, 0.3), p=1.0)
            ], p=0.2),

            A.OneOf([
                A.VerticalFlip(p=1.0),
                A.ThinPlateSpline(scale_range=(0.2, 0.4), num_control_points=4, keypoint_remapping_method="mask", p=1.0)
            ], p=0.2),

            A.Rotate(limit=(-45, 45), border_mode=cv2.BORDER_REPLICATE, p=0.2),
            A.CropAndPad(percent=(-0.2, -0.2), p=0.2)
        ])
        return transform
        
    def augment_pair(self, images, label):

        def _augment(img0, img1, lab):
            # Force conversion to NumPy arrays
            img0 = np.asarray(img0)
            img1 = np.asarray(img1)
            # Convert lab to a Python scalar if needed.
            lab_val = lab.item() if hasattr(lab, "item") else lab

            # Debug print (optional)
            # print("Type of img0:", type(img0), "Shape:", img0.shape)
            if self.annotation_type == "angular":
                if lab_val > 0:
                    aug_img0 = self.transform(image=img0)["image"]
                    aug_img1 = img1 # self.transform(image=img1)["image"]
                else:
                    aug_img0 = img0
                    aug_img1 = img1
            else:    
                if lab_val < 1:
                    aug_img0 = self.transform(image=img0)["image"]
                    aug_img1 = img1 # self.transform(image=img1)["image"]
                else:
                    aug_img0 = img0
                    aug_img1 = img1


            return aug_img0, aug_img1, lab

        #Wrap the augmentation function with tf.py_function.
        aug_img0, aug_img1, lab = tf.py_function(
            func=_augment,
            inp=[images[0], images[1], label],
            Tout=[tf.float32, tf.float32, label.dtype]
        )
        # Set static shape information if available.
        aug_img0.set_shape(images[0].shape)
        aug_img1.set_shape(images[1].shape)

        return (aug_img0, aug_img1), label
        
    def createPairs(self, dataset_path):

        celeb_folders = os.listdir(dataset_path)

        # Positive pairs
        pos_pairs = []

        for folder in celeb_folders:
            images = glob.glob(os.path.join(dataset_path, folder, "*.jpg"))
            pos_pairs.extend(combinations(images, 2))

        pos_pairs = random.sample(pos_pairs, min(len(pos_pairs), self.max_pos_pairs))
        # pos_pairs = list(map(self.augment_pair, pos_pairs)) if self.apply_augmentation else pos_pairs

        num_pos_pairs = len(pos_pairs)

        # Negative pair generator (no folder flip, so no (img2, img1) duplicates)
        def negative_pairs():
            for folderA, folderB in combinations(celeb_folders, 2):
                imgsA = glob.glob(os.path.join(dataset_path, folderA, "*.jpg"))
                imgsB = glob.glob(os.path.join(dataset_path, folderB, "*.jpg"))
                for imgA in imgsA:
                    for imgB in imgsB:
                        yield (imgA, imgB)

        # Reservoir-sample exactly num_pos_pairs negatives without storing them all
        neg_pairs = self.reservoir_sample(negative_pairs(), num_pos_pairs)

        print(f"Number of positive pairs: {len(pos_pairs)}")
        print(f"Number of negative pairs: {len(neg_pairs)}")

        return pos_pairs, neg_pairs
        
    def reservoir_sample(self, generator, k):
        """Keep a random sample of k items from the given generator/iterable."""
        sample = []
        n = 0
        for item in generator:
            n += 1
            if n <= k:
                sample.append(item)
            else:
                r = random.randint(1, n)
                if r <= k:
                    sample[r - 1] = item
        return sample
        
    def createTfDataset(self, pos_pairs, neg_pairs, image_shape, batch_size):

        x_pos, x_neg = tf.convert_to_tensor(pos_pairs), tf.convert_to_tensor(neg_pairs)

        if self.annotation_type == None:
            y_pos, y_neg = tf.zeros(len(pos_pairs), 1), tf.ones(len(neg_pairs), 1)

        if self.annotation_type=="angular":
            y_pos, y_neg = tf.ones(len(pos_pairs), 1), tf.zeros(len(pos_pairs), 1)

        X = tf.concat([x_pos, x_neg], axis=0)
        Y = tf.concat([y_pos, y_neg], axis = 0)

        dataset = tf.data.Dataset.from_tensor_slices((X, Y))
        dataset = dataset.map(lambda paths, label: self.load_and_preprocess_images(paths, label, image_shape))

        dataset = dataset.map(lambda images, label: self.augment_pair(images, label)) if self.apply_augmentation else dataset

        dataset = dataset.shuffle(len(pos_pairs) * 2)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size = 2)
    
        return dataset
    
    @tf.function
    def process_file(self, file_path, image_shape):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_image(img, channels=3)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, (image_shape[0], image_shape[1]))
        img = tf.ensure_shape(img, image_shape)
        
        if self.preprocess_pipeline:
            img = self.preprocess_pipeline(img)
            
        else: 
            img = tf.image.rgb_to_grayscale(img) if image_shape[-1] == 1 else img
            img = tf.cast(img, tf.float32)
            img = img / 255.0 if self.normalize else img
            
        return img

    def load_and_preprocess_images(self, image_paths, label, image_shape):
        # Process both images from the pair
        image1 = self.process_file(image_paths[0], image_shape)
        image2 = self.process_file(image_paths[1], image_shape)
        return (image1, image2), label  # Return as a tuple
    
    def visualize(self, value_range, color_mode_switch):
        
        vmin, vmax = value_range
        
        num_display = 20  # number of images to display
        _, axes = plt.subplots(nrows=num_display, ncols=2, figsize=(15, 5 * num_display))

        for element in self.dataset.take(1):
            (image1, image2), label = element

            # Convert tensors to numpy arrays without resizing
            image1_np = image1.numpy()
            image2_np = image2.numpy()
            labels_np = label.numpy()

            # Slice only the first num_display images from the batch
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
        
    @tf.function
    def process_file_embedding(self, file_path, label):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_image(img, channels=3)
        img.set_shape([None, None, 3])
        img = tf.image.rgb_to_grayscale(img) if self.image_shape[-1] == 1 else img
        img = tf.image.resize(img, (self.image_shape[0], self.image_shape[1]))
        img = tf.ensure_shape(img, self.image_shape)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label


    def create_embedding_dataset(self):
    
        image_paths, labels, class_names = self.get_image_paths_labels()

        image_paths = tf.constant(image_paths)
        labels = tf.constant(labels, dtype=tf.int32)

    
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(lambda file_path, label: self.process_file_embedding(file_path, label),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset, class_names
        
        
        
         
       
        
       
       
    
        
        
    