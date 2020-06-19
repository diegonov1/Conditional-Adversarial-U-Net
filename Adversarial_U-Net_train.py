#This version is a draft and is currently being revised. It is not the final version.

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#Main file-path
PATH = 'Term_Paper/'

#input data path
INPATH = PATH + 'VOC_DATA/voc_input'

#target data path
OUPATH = PATH + 'VOC_DATA/voc_target'

#checkpoints data path
CKPATH = PATH + 'CHECKPOINT'

#GCP Console Command
image_list = !ls -1 '{INPATH}' #all in one column

n = 500
train_n = round(n * 0.80)

#Randomized lists
rand_list = np.copy(image_list)
np.random.seed(23)
np.random.shuffle(rand_list)

#Train/Test Split
tr_urls = rand_list[:train_n]
ts_urls = rand_list[train_n:n]

print(len(image_list), len(tr_urls), len(ts_urls))

#Fixing image size

IMG_WIDTH = 500
IMG_HEIGH = 500

#resizing the images function
def resize(input_img, target_img, height, width):
    input_img = tf.image.resize(input_img, [height, width])
    target_img = tf.image.resize(target_img, [height, width])
    
    return input_img, target_img

#normalize pictures function 
def normalize(input_img, target_img):
    input_img = (input_img / 127.5) - 1
    target_img = (input_img / 127.5) - 1
    
    return input_img, target_img

#Data augmentation function (Crop and Flip)
def random_jitter(input_img, target_img):
    input_img, target_img = resize(input_img, target_img, 700, 700)
    #stacking function
    stacked_image = tf.stack([input_img, target_img], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGH, IMG_WIDTH, 3])
    
    input_img, target_img = cropped_image[0], cropped_image[1]
    
    if tf.random.uniform(()) > 0.5:
        
        input_img = tf.image.flip_left_right(input_img)
        target_img = tf.image.flip_left_right(target_img)
        
    return input_img, target_img

def load_image(filename, augment=True):
    
    input_img = tf.cast(tf.image.decode_jpeg(tf.io.read_file(INPATH + '/' + filename)), tf.float32)[..., :3]
    reimg = tf.cast(tf.image.decode_jpeg(tf.io.read_file(OUPATH + '/' + filename)), tf.float32)[..., :3]

    input_img, reimg = resize(input_img, reimg, IMG_HEIGH, IMG_WIDTH)
    
    if augment:
      input_img, reimg = random_jitter(input_img, reimg)
        
    input_img, reimg = normalize(input_img, reimg)
    
    return input_img, reimg

def load_train_image(filename):
    return load_image(filename, True)

def load_test_image(filename):
    return load_image(filename, False)

from tensorflow.keras.layers import *
from tensorflow.keras import Sequential, Model

def downSample(filters):
    
    result = Sequential()
    initializer = tf.random_normal_initializer(0, 0.02)
    
    #Convolutional Layer
    result.add(Conv2D(filters,
                      kernel_size=4,
                      strides=2,
                      padding='same',
                      kernel_initializer=initializer,
                      use_bias=not apply_batchnorm))
    #BatchNorm Layer
    if apply_batchnorm:
        result.add(BatchNormalization())
     
    #Activation
    result.add(LeakyReLU)
    
    return result

def upsample(filters, apply_dropout=True):
    
    result = Sequential()
    initializer = tf.random_normal_initializer(0, 0.02)
    
    #Convolutional Layer
    result.add(Conv2DTranspose(filters,
                               kernel_size=4,
                               strides=2,
                               padding='same',
                               kernel_initializer=initializer,
                               use_bias=False))
    #BatchNorm Layer
    result.add(BatchNormalization())
    
    if apply_dropout:
        result.add(Dropout(0.5))
     
    #Activation
    result.add(ReLU)
    
    return result

def generator():
    inputs = tf.keras.layers.Input(shape=[None, None, 3])
    
    down_stack = [
        downsample(64, apply_batchnorm=False),
        downsample(128),
        downsample(256),
        downsample(512),
        downsample(512),
        downsample(512),
        downsample(512),
        downsample(512),
    ]
    
    up_stack = [
        upsample(512, apply_dropout=True),
        upsample(512, apply_dropout=True),
        upsample(512, apply_dropout=True),
        upsample(512),
        upsample(256),
        upsample(128),
        upsample(64),
    ]
    
    initializer = tf.random_normal_initializer(0, 0.02)
    
    last = Conv2DTranspose(filters=3,
                          kernel_size=4,
                          strides=2,
                          padding='same',
                          kernel_initializer = initializer,
                          activation='tanh')
    
    x = inputs
    skip_x = []
    concat = Concatenate()
    
    for down in down_stack:
        x = down(x)
        skip_x.append(x)
    
    skip_x = reversed(skip_x[:-1])
    
    for up, sk in zip(up_stack, skip_x):
        x = up(x)
        x = concat([x, sk])
    
    last = last(x)
    
    return Model(inputs=inputs, outputs=last)

def discriminator():
    
    initializer = tf.random_normal_initializer(0, 0.02)
    
    ini = Input(shape=[None, None, 3], name='input_img')
    gen = Input(shape=[None, None, 3], name='gen_img')

    con = concatenate([ini, gen])

    down1 = downsample(64, apply_batchnorm=False)(con)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)
    down4 = downsample(256, 4)(down3)
    
    last = Conv2D(filters=1,
                  kernel_size=4,
                  strides=1,
                  padding='same',
                  kernel_initializer=initializer)(down4)
    
    return Model(input=[ini, gen], outputs=last)

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True) #logits True = 0 to 1

def discriminatorLoss(disc_real_output, disc_generated_output):
    #Real true vs detected by discriminator
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    #Real false vs detected by discriminator
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    
    total_disc_loss = real_loss + generated_loss
    
    return total_disc_loss

lamda = 100

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    #Mean Absolute Error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    
    total_gen_loss = gan_loss + l1_loss
    
    return total_gen_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

import os

checkpoint_prefix = os.path.join(CKPATH, 'ckpt')
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                generator=generator,
                                discriminator=discriminator)

#checkpoint.restore(tf.train.latest_checkpoint(CKPATH)).assert_consumed()

def generate_images(model, test_input, tar, save_filename=False, display_imgs=True):
    
    prediction = model(test_input, training=True)
    
    if save_filename:
        tf.keras.preprocessing.image.save_img(PATH + 'OUTPUT' + save_filename + '.jpg', prediction[0, ...])
        
    plt.figure(figsize=(10,10))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    
    if display_imgs:
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
    plt.show()

@tf.function()
def train_step(input_image, target):
    
    with tf.GradientTape() as gen_tape, tf.GradientTape as discr_tape:
    
        output_image = generator(input_image, training=Ture)
        output_gen_discr = discriminator([output_image, input_image], training=True)
        output_trg_discr = discriminator([target, input_image], training=True)
        discr_loss = discriminator_loss(output_trg_discr, output_gen_discr)
        gen_loss = generator_loss(output_gen_discr, output_image, target)
        
        generator_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
        discriminator_grads = discr_tape.gradient(discr_loss, discriminator.trainable_variables)
        generator_optimizer.apply_gradients(zip(generator_grads, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_grads, discriminator.trainable_variables))

from IPython.display import clear_output

def train(dataset, epochs):
    for epoch in range(epochs):
        img_i = 0
        for input_image, target in dataset:
            print('Epoch:'  + str(epoch) + ' - Train: ' + str(img_i) + '/' + str(len(tr_urls)))
            img_i += 1
            train_step(input_image, target)
        
            clear_outputs(wait=True)
        
        for inp, tar in test_dataset.take(5):
            generate_images(generator, inp, tar, str(img_i) + '_' + str(epoch), display_imgs=True)
            
        if (epoch + 1) % 25 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

train(train_dataset, 100)