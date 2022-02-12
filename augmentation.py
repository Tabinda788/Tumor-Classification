import os
import zipfile
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
from os import getcwd
from os import listdir
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image  as mpimg



def augment_data(file_dir, n_generated_samples, save_to_dir):
    #from keras.preprocessing.image import ImageDataGenerator
    #from os import listdir
    
    data_gen = ImageDataGenerator(rotation_range=10, 
                                  width_shift_range=0.1, 
                                  height_shift_range=0.1, 
                                  shear_range=0.1, 
                                  brightness_range=(0.3, 1.0),
                                  horizontal_flip=True, 
                                  vertical_flip=True, 
                                  fill_mode='nearest'
                                 )

    
    for filename in listdir(file_dir):
        # load the image
        image = cv2.imread(file_dir + '\\' + filename)
        # reshape the image
        image = image.reshape((1,)+image.shape)
        # prefix of the names for the generated sampels.
        save_prefix = 'aug_' + filename[:-4]
        # generate 'n_generated_samples' sample images
        i=0
        for batch in data_gen.flow(x=image, batch_size=1, save_to_dir=save_to_dir, 
                                           save_prefix=save_prefix, save_format='jpg'):
            i += 1
            if i > n_generated_samples:
                break



augmented_data_path = 'C:\\Users\\adeel\\OneDrive\\Desktop\\Tumor-Classification\\Tumor-Classification'

# augment data for the examples with label equal to 'yes' representing tumurous examples
augment_data(file_dir='yes', n_generated_samples=6, save_to_dir=augmented_data_path+'yesreal')
# augment data for the examples with label equal to 'no' representing non-tumurous examples
augment_data(file_dir='no', n_generated_samples=9, save_to_dir=augmented_data_path+'noreal')


def data_summary(main_path):
    
    yes_path = main_path+'yesreal'
    print("printing yes path",yes_path)
    no_path = main_path+'noreal'
        
    # number of files (images) that are in the the folder named 'yes' that represent tumorous (positive) examples
    m_pos = len(listdir(yes_path))
    m_neg = len(listdir(no_path))
    m = (m_pos+m_neg)
    
    pos_prec = (m_pos* 100.0)/ m
    neg_prec = (m_neg* 100.0)/ m
    
    print(f"Number of examples: {m}")
    print(f"Percentage of positive examples: {pos_prec}%, number of pos examples: {m_pos}") 
    print(f"Percentage of negative examples: {neg_prec}%, number of neg examples: {m_neg}") 
    
    
data_summary(augmented_data_path)
