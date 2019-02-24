import numpy as np
import time
from keras import backend as K
from keras.applications import vgg16
from keras.preprocessing.image import load_img, img_to_array


def time_taken(f, *args):
    start_time = time.time()
    f(*args)
    end_time = time.time()
    return end_time - start_time


# utility functn to open, resize, format images into appropriate tensors
def preprocess_image(image_path, dimensions):
    image = load_img(image_path, target_size=dimensions)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = vgg16.preprocess_input(image)
    return image


# utility fucntion to convert a tensor into a valid image
def deprocess_image(x, image_rows, image_columns):
    if K.image_dim_ordering() == 'th':
        x = x.reshape((3, image_rows, image_columns))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((image_rows, image_columns, 3))

    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x