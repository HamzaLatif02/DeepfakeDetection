# adapted from resnet30, added a third layer in identity and convolutional block to respect architecture

import tensorflow as tf

def identity_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # layer 1
    x = tf.keras.layers.Conv2D(filter, (1,1), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # layer 2
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # layer 3
    x = tf.keras.layers.Conv2D(filter*4, (1,1), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # ensure x_skip has the same number of filters
    if x.shape[-1] != x_skip.shape[-1]:
        x_skip = tf.keras.layers.Conv2D(filter*4, (1,1), padding='same')(x_skip)
        x_skip = tf.keras.layers.BatchNormalization(axis=3)(x_skip)
    # add Residue
    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)
    return x

def convolutional_block(x, filter):
    x_skip = x
    # layer 1
    x = tf.keras.layers.Conv2D(filter, (1,1), padding='same', strides=(2,2))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # layer 2
    x = tf.keras.layers.Conv2D(filter, (3,3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # layer 3
    x = tf.keras.layers.Conv2D(filter*4, (1,1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # adjusting x_skip to match the filter*4 and applying stride to match dimension reduction
    x_skip = tf.keras.layers.Conv2D(filter*4, (1,1), padding='same', strides=(2,2))(x_skip)
    x_skip = tf.keras.layers.BatchNormalization(axis=3)(x_skip)
    # add the main path and the shortcut together
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def ResNet50(shape = (256, 256, 3), classes = 10):
    # setup input layer
    x_input = tf.keras.layers.Input(shape)
    x = tf.keras.layers.ZeroPadding2D((3, 3))(x_input)
    # initial conv layer along with maxPool
    x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    # define size of sub-blocks and initial filter size
    block_layers = [3, 4, 6, 3]
    filter_size = 64
    # add the Resnet Blocks
    for i in range(4):
        if i == 0:
            # for sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[i]):
                x = identity_block(x, filter_size)
        else:
            # one Residual/Convolutional Block followed by Identity blocks
            # the filter size will go on increasing by a factor of 2
            filter_size = filter_size*2
            x = convolutional_block(x, filter_size)
            for j in range(block_layers[i] - 1):
                x = identity_block(x, filter_size)
    # end Dense Network
    x = tf.keras.layers.AveragePooling2D((2,2), padding = 'same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation = 'relu')(x)
    x = tf.keras.layers.Dense(classes, activation = 'softmax')(x)
    model = tf.keras.models.Model(inputs = x_input, outputs = x, name = "ResNet50")
    return model