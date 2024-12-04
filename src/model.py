import tensorflow as tf
import keras
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout 
from keras.layers import BatchNormalization

from keras.layers import Conv2DTranspose
from keras.layers import concatenate

# Define model architecture, same as model described in Zambrano et al
def load_c2c_model():
    n_filters=32
    n_classes = 5
    inputs = Input((512,512,3))
    cblock1 = conv_block(inputs, n_filters)
    cblock2 = conv_block(cblock1[0], n_filters*2)
    cblock3 = conv_block(cblock2[0], n_filters*4)
    cblock4 = conv_block(cblock3[0], n_filters*8)
    cblock5 = conv_block(cblock4[0], n_filters*16)
    cblock6 = conv_block(cblock5[0], n_filters*32,max_pooling=False)

    ublock7 = upsampling_block(cblock6[0], cblock5[1], n_filters*16)
    ublock8 = upsampling_block(ublock7, cblock4[1], n_filters*8)
    ublock9 = upsampling_block(ublock8, cblock3[1], n_filters*4)
    ublock10 = upsampling_block(ublock9, cblock2[1], n_filters*2)
    ublock11 = upsampling_block(ublock10, cblock1[1], n_filters)

    conv12 = Conv2D(n_classes, 1, padding='same', activation='softmax')(ublock11)
    model = tf.keras.Model(inputs=inputs, outputs=conv12)
    # model.summary()

    # Initialize weights from C2C
    model.set_weights(c2c_weights)

    # Freeze Model Weights
    model.trainable = False

    # Set output layer to 9 classes 
    num_classes_muscle = 9
    new_output = Conv2D(num_classes_muscle, 1, padding='same', activation='softmax', kernel_initializer='he_normal', trainable=True)(model.layers[-2].output)
    model = tf.keras.Model(inputs=model.input, outputs=new_output)
    return model



def conv_block(inputs=None, n_filters=32, dropout_prob=0.3, max_pooling=True): #Add trainability setting?
    """
    Convolutional downsampling block
    
    Arguments:
        inputs -- Input tensor
        n_filters -- Number of filters for the convolutional layers
        dropout_prob -- Dropout probability
        max_pooling -- Use MaxPooling2D to reduce the spatial dimensions of the output volume
    Returns: 
        next_layer, skip_connection --  Next layer and skip connection outputs
    """

    conv = Conv2D(n_filters, #number of filters
                  3, # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(inputs)
    
    conv = Conv2D(n_filters, #number of filters
                  3, # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(conv)

    # Batch Normalizatiion
    conv = BatchNormalization(
        axis=-1,
        momentum=0.99,
        epsilon=0.001
    )(conv)
    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)
    
    if max_pooling:
        next_layer = MaxPooling2D(pool_size=(2,2))(conv)
    
    else:
        next_layer = conv
    
    skip_connection = conv

    return next_layer, skip_connection


def upsampling_block(expansive_input, contractive_input, n_filters=32, dropout_prob=0.3):
    """
    Convolutional upsampling block
    
    Arguments:
        expansive_input -- Input tensor from previous layer
        contractive_input -- Input tensor from previous skip layer
        n_filters -- Number of filters for the convolutional layers
    Returns: 
        conv -- Tensor output
    """

    up = Conv2DTranspose(n_filters,
                         3,
                         strides=(2,2),
                         padding='same')(expansive_input)

    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(n_filters,
                  3, 
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(merge)

    conv = Conv2D(n_filters,
                  3, 
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(conv)
    
    # Batch Normalizatiion
    conv = BatchNormalization(
        axis=-1,
        momentum=0.99,
        epsilon=0.001,
    )(conv)
    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)

    return conv