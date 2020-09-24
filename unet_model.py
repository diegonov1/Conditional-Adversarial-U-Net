import tensorflow as tf
import hyperparameters

def downsample(in_layer, filters):
    
    conv1 = tf.keras.layers.Conv2D(filters, (3, 3), 
                                   padding='same',
                                   kernel_initializer='glorot_uniform',
                                   activation='relu')(in_layer)

    conv2 = tf.keras.layers.Conv2D(filters, (3, 3), 
                                   padding='same',
                                   kernel_initializer='glorot_uniform',
                                   activation='relu')(conv1)
    
    bnorm = tf.keras.layers.BatchNormalization()(conv2)

    maxpool = tf.keras.layers.MaxPool2D()(conv2)

    return maxpool, bnorm


def upsample(in_layer, filters):

    conv1 = tf.keras.layers.Conv2DTranspose(filters*2, (3, 3),
                                            padding='same',
                                            kernel_initializer='glorot_uniform',
                                            activation='relu')(in_layer)

    conv2 = tf.keras.layers.Conv2DTranspose(filters*2, (3, 3),
                                            padding='same',
                                            kernel_initializer='glorot_uniform',
                                            activation='relu')(conv1)

    bnorm = tf.keras.layers.BatchNormalization()(conv2)

    upsample = tf.keras.layers.UpSampling2D()(bnorm)

    return upsample


def Unet_Generator():

    inputs = tf.keras.layers.Input(shape=[hyperparameters.IMG_WIDTH, hyperparameters.IMG_HEIGHT, hyperparameters.OUTPUT_CHANNELS])

    last1 = tf.keras.layers.Conv2DTranspose(64, 3,
                                            padding='same',
                                            kernel_initializer='glorot_uniform',
                                            activation='relu') # (bs, 256, 256, 3)

    last2 = tf.keras.layers.Conv2DTranspose(hyperparameters.OUTPUT_CHANNELS, 3,
                                            padding='same',
                                            kernel_initializer='glorot_uniform',
                                            activation='tanh') # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model

    skips = []
    filter_down = [64, 128, 256, 512, 512]

    for filt in filter_down:
        x, y = downsample(x, filt)
        skips.append(y)

    skips = reversed(skips)

    filter_up = [512, 512, 256, 128, 64]

    # Upsampling and establishing the skip connections
    for filt, skip in zip(filter_up, skips):
        
        x = upsample(x, filt)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last1(x)
    x = last2(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (hyperparameters.LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss