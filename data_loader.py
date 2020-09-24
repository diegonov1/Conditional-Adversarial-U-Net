import tensorflow as tf
import data_augmentation
import hyperparameters
import os

def load_image(filename, augment=True):
  
    train_img = tf.cast(tf.image.decode_jpeg(tf.io.read_file(TRAIN_IMG_PATH + '/' + filename)), tf.float32)
    masked_img = tf.cast(tf.image.decode_jpeg(tf.io.read_file(TRAIN_MSK_PATH + '/' + filename)), tf.float32)

    train_img, masked_img = data_augmentation.resize(train_img, masked_img, hyperparameters.IMG_HEIGHT, 
                                                     hyperparameters.IMG_WIDTH)
    
    if augment:
        train_img, masked_img = data_augmentation.randomizer(train_img, masked_img)
      
    train_img, masked_img = data_augmentation.normalize(train_img, masked_img)
    
    return train_img, masked_img

def load_train_img(filename):
    return load_image(filename, True)

def load_validation_img(filename):
    return load_image(filename, False)

train_dataset = tf.data.Dataset.from_tensor_slices(train_imgs)
train_dataset = train_dataset.map(data_loader.load_train_img)
train_dataset = train_dataset.shuffle(hyperparameters.BUFFER_SIZE)
train_dataset = train_dataset.batch(hyperparameters.BATCH_SIZE)

validation_dataset = tf.data.Dataset.from_tensor_slices(validation_imgs)
validation_dataset = validation_dataset.map(data_loader.load_validation_img)
validation_dataset = validation_dataset.batch(hyperparameters.BATCH_SIZE)