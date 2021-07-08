import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
import numpy as np

def conv(x,filters):
    x = layers.Conv1D(filters,kernel_size=1,padding='valid')(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation('relu')(x)

def dens(x,filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation('relu')(x)


def tnet(inputs, num_features):
    """
    This is the core t-net of the pointnet paper
    :param inputs: the input tensor
    :type inputs: tensor
    :param num_features: number of features in the tensor (3 for point cloud, N if N features)
    :type num_features: int
    :return: output tensor
    :rtype: tensor
    """

    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    
    dims = inputs.shape
    print(dims)# 1 x 1024 x 3
    # TODO: Build the tnet with the following layers
    # Some convolutional layers (1D) - with batch normalization, RELU activation
    x = conv(inputs,64)
    # 1 x 1024 x 64
    x = conv(x,128)
    # 1 x 1024 x 128
    x = conv(x,1024)
    # 1 x 1024 x 1024
    # Global max pooling
    x = layers.GlobalMaxPool1D()(x)
    # 1 x 1024 x 1
    # Some dense fully connected layers - with batch normalization, RELU activation
<<<<<<< HEAD
    x = dens(x,512)
    x = dens(x,256)


=======
    # final layer with custom regularizer on the output
    # TODO: this custom regularizer needs to be defined
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=CustomRegularizer(num_features)
    )(x)

    feat_t = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_t])


class CustomRegularizer(keras.regularizers.Regularizer):
    """
    This class implements a regularizer that makes the output to be orthogonal.
    In other words, it adds a loss |I-AA^T|^2 on the output A. Equation 2 of the paper.
    """

    def __init__(self, dim, weight=0.001):
        """
        Initializes the class
        :param dim: dimensions of the input tensor
        :type dim: int
        :param weight: weight to apply on the regularizer
        :type weight: float
        """
        self.dim = dim
        self.weight = weight
        self.eye = tf.eye(dim)

    def __call__(self, x):
        # TODO: define the custom regularizer here
        x = tf.reshape(x, (-1, self.dim, self.dim))
        # compute the outer product and reshape it to batch size x num_features x num_features
        outerpr = tf.tensordot(x, tf.transpose(x), axes=(2,2))
        outerpr = tf.reshape(outerpr, (-1, self.dim, self.dim))     # use .reshape(self.dim) ??
        # Compute (I-outerproduct)^2 element wise. use tf.square()
        out = tf.square(self.eye - outerpr)         # use self.dim here
        # Apply weight
        out = self.weight * out
        # Compute reduce sum using tf.reduce_sum()
        output = tf.reduce_sum(out)
        return output


def pointnet_classifier(inputs, num_classes):
    """
    This is the object classifier version of PointNet
    :param inputs: input point clouds tensor
    :type inputs: tensor
    :param num_classes: number of classes
    :type num_classes: int
    :return: the predicted labels
    :rtype: tensor
    """
    # TODO: build the network using the following layers
    # apply tnet to the input data
<<<<<<< HEAD
    x = tnet(inputs, 3)

    # extract features using some Convolutional Layers - with batch normalization and RELU activation
    x = conv(x,32)
    x = conv(x,64)
    
    # apply tnet on the feature vector
    x = tnet(x, 64)
    # TODO: Check dimension mismatch?

    # extract features using some Convolutional Layers - with batch normalization and RELU activation
    x = conv(x, 128)
    x = conv(x, 256)
    x = conv(x, 1024)
    # apply 1D global max pooling
    x = layers.GlobalMaxPool1D()(x)

    # Add a few dense layers with dropout between the layers
    x = dens(x,512)
    x = layers.Dropout(0.3)(x)      # TODO: should this be 0.7?
    x = dens(x, 256)
    x = layers.Dropout(0.3)(x)      # TODO: should this be 0.7?
    
    # Finally predict classes using a dense layer with a softmax activation
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return outputs


def pointnet_segmenter(inputs, labels):
    """
    This is the semantic segmentation version of Pointnet
    :param inputs: input point cloud
    :type inputs: tensor
    :param labels: labels for each point of the point cloud
    :type labels: tensor
    :return: predicted labels for each point of the point cloud
    :rtype: tensor
    """
    # TODO: build the network using the following layers
    # apply tnet to the input data
<<<<<<< HEAD
    x = tnet(inputs, 3)
>>>>>>> 7b8c14948690ea97daba94e031004e4adc38c012
    # extract features using some Convolutional Layers - with batch normalization and RELU activation
    x = conv(x,32)
    x = conv(x,64)
    # apply tnet on the feature vector
    # f =
    # extract features using some Convolutional Layers - with batch normalization and RELU activation

    # apply 1D global max pooling

    # concatenate these features with the earlier features (f)
    # you can also use skip connections if you like

    # extract features using some Convolutional Layers - with batch normalization and RELU activation

    # return the output
    return outputs

