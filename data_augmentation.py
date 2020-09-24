import tensorflow as tf
import hyperparameters

def resize(train_img, masked_img, height, width):

    train_img = tf.image.resize(train_img, [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    masked_img = tf.image.resize(masked_img, [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return train_img, masked_img


def random_crop(train_img, masked_img):

    stacked_image = tf.stack([train_img, masked_img], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, hyperparameters.IMG_HEIGHT, hyperparameters.IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]


def normalize(train_img, masked_img):

    train_img = (train_img / 127.5) - 1
    masked_img = (masked_img / 127.5) - 1

    return train_img, masked_img


@tf.function()
def randomizer(train_img, masked_img):

    train_img, masked_img = resize(train_img, masked_img, hyperparameters.IMG_WIDTH+50, hyperparameters.IMG_HEIGHT+50)

    train_img, masked_img = random_crop(train_img, masked_img)

    if tf.random.uniform(()) > 0.5:

        train_img = tf.image.flip_left_right(train_img)
        masked_img = tf.image.flip_left_right(masked_img)

    return train_img, masked_img
