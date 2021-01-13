import os
import numpy as np
from PIL import Image
import scipy.ndimage

import tensorflow.keras.datasets as dataset_utils

from absl import logging as logger

# Initialise logging globally
logger.set_verbosity('info')
logger.set_stderrthreshold('info')



class ColoredMNIST:
    
    def __init__(self, root='./dataset', base_image = 'background.jpg'):
        
        self.root = root 
        self.base_image = base_image
        self.train_size = None 
        self.test_size = None
        
        self.train_dataset = None 
        self.test_dataset = None
        
        self.generate_colored_mnist()
        
    def load_data(self):
        
        return (self.train_dataset, self.test_dataset)
        
    def get_color_image(self):
        
        base = Image.open(os.path.join(self.root, self.base_image))
        return base 
    
    @staticmethod
    def crop_image_batch(image, batch_size):
        
        base_image_batch = []
        
        for i in range(batch_size):
            x_crop = np.random.randint(0, image.size[0] - 64)
            y_crop = np.random.randint(0, image.size[1] - 64)
            # crop image by size 64*64 
            cropped_image = image.crop((x_crop, y_crop, x_crop+64, y_crop+64))
            # normalize image value 
            cropped_normalized_image = np.asarray(cropped_image) / 255.0
            
            base_image_batch.append(cropped_normalized_image)
        
        return base_image_batch 
        
    def generate_colored_mnist(self):
        
        colored_mnist_dir = os.path.join(self.root, 'ColoredMNIST')
        
        logger.info('Preparing Colored MNIST')
        
        mnist = dataset_utils.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # downsize 
        x_train = x_train[:5000]
        y_train = y_train[:5000]
        x_test = x_test[:500]
        y_test = y_test[:500]
        # normalise 
        x_train, x_test = x_train / 255.0, x_test / 255.0
        # Get data size 
        self.train_size = x_train.shape[0] 
        self.test_size = x_test.shape[0] 
        
        # Add color channel 
        x_train_rgb = x_train.reshape(-1, 28, 28, 1).astype(np.float32)
        x_test_rgb = x_test.reshape(-1, 28, 28, 1).astype(np.float32)
        
        # Scale image from 28*28 to 64*64  
        x_train_resized = np.asarray([scipy.ndimage.zoom(image, (2.3, 2.3, 1), order=1) for image in x_train_rgb])
        x_test_resized = np.asarray([scipy.ndimage.zoom(image, (2.3, 2.3, 1), order=1) for image in x_test_rgb])
        logger.info(f"shape of resized image: {x_train_resized[0].shape}")
        
        # Concetenate 3 single channel images to form RGB(3 channels) image 
        grey_mnist_train = np.concatenate([x_train_resized, x_train_resized, x_train_resized], axis=3)
        grey_mnist_test = np.concatenate([x_test_resized, x_test_resized, x_test_resized], axis=3)
        
        # Get binary image (True for pixel with hand written digit and False for blank pixel)
        binary_mnist_train = (grey_mnist_train > 0.5)
        binary_mnist_test = (grey_mnist_test > 0.5) 
        
        # Get base image 
        base_image = self.get_color_image()
        # Prepare batch of cropped base image 
        batch_size = self.train_size + self.test_size
        base_image_batch = self.crop_image_batch(base_image, batch_size)
        
        colored_train_set = []
        colored_test_set = [] 
        
        logger.info('Start overwirting cropped base image')
        for idx, binary_image in enumerate(binary_mnist_train):
            base_image = base_image_batch[idx]
            # Set the color of the hand written digit as White 
            base_image[binary_image] = 1 
            
            colored_train_set.append(base_image)
            
        for idx, binary_image in enumerate(binary_mnist_test):
            base_image = base_image_batch[self.train_size + idx]
            base_image[binary_image] = 1 
            colored_test_set.append(base_image)
            
        # Convert list of numpy (colored mnist image) to numpy 
        colored_mnist_array_tr = np.stack(colored_train_set, axis=0)
        colored_mnist_array_te = np.stack(colored_test_set, axis=0)
        
        logger.info(f"Shape of ColoredMNIST training set: {colored_mnist_array_tr.shape}")
        logger.info(f"Shape of ColoredMNIST test set: {colored_mnist_array_te.shape}")
        
        # Save to class instance
        self.train_dataset = (colored_mnist_array_tr, y_train) 
        self.test_dataset  = (colored_mnist_array_te, y_test) 
        
        # Save to /dataset directory 
        # Create directory if not exist 
#         if not os.path.isdir(colored_mnist_dir):
#             os.makedirs(colored_mnist_dir)

#         np.savez(os.path.join(colored_mnist_dir, 'train.npz'), image=colored_mnist_array_tr, label=y_train)
#         np.savez(os.path.join(colored_mnist_dir, 'test.npz'), image=colored_mnist_array_te, label=y_test)        
        
    
    