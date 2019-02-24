from keras import backend as K

def feature_content_loss(content, output):
    """
    Featue Representation loss function
    Encourages the output image to matche the feature responses of the original
    image
    content and output are feature representations of content image and output
    image respectively
    """
    return K.sum(K.square(output - content))


def gram_matrix(x):
    """
    The feature correlations are given by the
    Gram matrix, where G(l)ij is the inner product
    between the vectorised feature map i and j in layer l
    """
    if K.image_dim_ordering() == 'th':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))

    return K.dot(features, K.transpose(features))


def style_reconstruction_loss(style, output, image_rows, image_cols):
    """
    Style Reconstruction loss.Encourages to find another image that matches the style
    representation of the original image. This is done by minimising the mean-squared distance
    between the entries of the Gram matrix from the original image and the Gram matrix of the
    image to be generated.
    """
    h, w, c = image_rows, image_cols, 3
    fac = (1.0) / float((2 * h * w * c) ** 2)
    loss = fac * K.sum(K.square(gram_matrix(output) - gram_matrix(style)))
    return loss


def variation_loss(x, img_nrows, img_ncols):
    """
    Total variational loss. Encourages spatial smoothness
    in the output image.
    """
    H, W = img_nrows, img_ncols
    if K.image_dim_ordering() == 'th':
        a = K.square(x[:, :, :H - 1, :W - 1] - x[:, :, 1:, :W - 1])
        b = K.square(x[:, :, :H - 1, :W - 1] - x[:, :, :H - 1, 1:])
    else:
        a = K.square(x[:, :H - 1, :W - 1, :] - x[:, 1:, :W - 1, :])
        b = K.square(x[:, :H - 1, :W - 1, :] - x[:, :H - 1, 1:, :])

    return K.sum(K.pow(a + b, 1.25))
