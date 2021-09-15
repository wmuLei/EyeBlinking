
import keras
import tensorflow as tf

from keras import layers
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Activation, Subtract
from keras.layers import merge, BatchNormalization

from keras.models import Model
from keras import backend as K
from keras.layers.core import Lambda



def CONV2D(x, filter_num, kernel_size, activation='relu', **kwargs):
    x = Conv2D(filter_num, kernel_size, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    if activation=='relu': 
        x = Activation('relu', **kwargs)(x)
    elif activation=='sigmoid': 
        x = Activation('sigmoid', **kwargs)(x)
    else:
        x = Activation('softmax', **kwargs)(x)
    return x



def TFnet(shape, classes=1):
    inputs = Input(shape)

    def interSubtract(x0, x1):
        x0 = Subtract()([x0, x1])
        x0 = Subtract()([Lambda(lambda x: K.abs(x))(x0), x0]);
        return x0

    def intraSubtract(x0, filter_num=32):
        x1 = Lambda(lambda x: x[:, :, :, 0:filter_num])(x0)
        x2 = Lambda(lambda x: x[:, :, :, filter_num: ])(x0)
        x0 = interSubtract(x1, x2)
        return x0
    
    conv0 = BatchNormalization()(inputs)
    conv0 = CONV2D(conv0, 32, (3, 3)); edge1 = intraSubtract(conv0, 16); edge1 = merge([conv0, edge1], mode='concat');
    conv1 = CONV2D(edge1, 32, (3, 3)); edge1 = intraSubtract(conv1, 16); edge1 = merge([conv1, edge1], mode='concat');
    conv0 = interSubtract(conv0, conv1);
    conv1 = merge([edge1, conv0], mode='concat');    
    conv0 = MaxPooling2D(pool_size=(2, 2))(conv1);  # size/2

    conv0 = CONV2D(conv0, 64, (3, 3)); edge1 = intraSubtract(conv0, 32); edge1 = merge([conv0, edge1], mode='concat');
    conv2 = CONV2D(edge1, 64, (3, 3)); edge1 = intraSubtract(conv2, 32); edge1 = merge([conv2, edge1], mode='concat');
    conv0 = interSubtract(conv0, conv2); 
    conv2 = merge([edge1, conv0], mode='concat');    
    conv0 = MaxPooling2D(pool_size=(2, 2))(conv2);  # size/4

    conv0 = CONV2D(conv0, 128, (3, 3)); edge1 = intraSubtract(conv0, 64); edge1 = merge([conv0, edge1], mode='concat');
    conv3 = CONV2D(edge1, 128, (3, 3)); edge1 = intraSubtract(conv3, 64); edge1 = merge([conv3, edge1], mode='concat');
    conv0 = interSubtract(conv0, conv3);  
    conv3 = merge([edge1, conv0], mode='concat');    
    conv0 = MaxPooling2D(pool_size=(2, 2))(conv3);  # size/8

    conv0 = CONV2D(conv0, 256, (3, 3)); edge1 = intraSubtract(conv0, 128); edge1 = merge([conv0, edge1], mode='concat');
    conv4 = CONV2D(edge1, 256, (3, 3)); edge1 = intraSubtract(conv4, 128); edge1 = merge([conv4, edge1], mode='concat');
    conv0 = interSubtract(conv0, conv4);
    conv4 = merge([edge1, conv0], mode='concat');    
    conv0 = MaxPooling2D(pool_size=(2, 2))(conv4);  # size/16

    conv0 = CONV2D(conv0, 512, (3, 3)); edge1 = intraSubtract(conv0, 256); edge1 = merge([conv0, edge1], mode='concat');
    conv5 = CONV2D(edge1, 512, (3, 3)); edge1 = intraSubtract(conv5, 256); edge1 = merge([conv5, edge1], mode='concat');
    conv0 = interSubtract(conv0, conv5);
    conv5 = merge([edge1, conv0], mode='concat');    
    conv0 = MaxPooling2D(pool_size=(2, 2))(conv5);  # size/32

    #----------------------------------------------
    conv0 = CONV2D(conv0, 1024, (3, 3)); edge1 = intraSubtract(conv0, 512); edge1 = merge([conv0, edge1], mode='concat');
    conv6 = CONV2D(edge1, 1024, (3, 3)); edge1 = intraSubtract(conv6, 512); edge1 = merge([conv6, edge1], mode='concat');
    conv0 = interSubtract(conv0, conv6);   
    conv0 = merge([edge1, conv0], mode='concat');  # size/32
    #----------------------------------------------

    conv0 = merge([UpSampling2D(size=(2, 2))(conv0), conv5], mode='concat') # size/16
    conv0 = CONV2D(conv0, 512, (3, 3)); edge1 = intraSubtract(conv0, 256); edge1 = merge([conv0, edge1], mode='concat');
    conv5 = CONV2D(edge1, 512, (3, 3)); edge1 = intraSubtract(conv5, 256); edge1 = merge([conv5, edge1], mode='concat');
    conv0 = interSubtract(conv0, conv5);
    conv0 = merge([edge1, conv0], mode='concat');


    conv0 = merge([UpSampling2D(size=(2, 2))(conv0), conv4], mode='concat') # size/8
    conv0 = CONV2D(conv0, 256, (3, 3)); edge1 = intraSubtract(conv0, 128); edge1 = merge([conv0, edge1], mode='concat');
    conv4 = CONV2D(edge1, 256, (3, 3)); edge1 = intraSubtract(conv4, 128); edge1 = merge([conv4, edge1], mode='concat');
    conv0 = interSubtract(conv0, conv4);
    conv0 = merge([edge1, conv0], mode='concat');


    conv0 = merge([UpSampling2D(size=(2, 2))(conv0), conv3], mode='concat') # size/4
    conv0 = CONV2D(conv0, 128, (3, 3)); edge1 = intraSubtract(conv0, 64); edge1 = merge([conv0, edge1], mode='concat');
    conv3 = CONV2D(edge1, 128, (3, 3)); edge1 = intraSubtract(conv3, 64); edge1 = merge([conv3, edge1], mode='concat');
    conv0 = interSubtract(conv0, conv3);  
    conv0 = merge([edge1, conv0], mode='concat');


    conv0 = merge([UpSampling2D(size=(2, 2))(conv0), conv2], mode='concat') # size/4
    conv0 = CONV2D(conv0, 64, (3, 3)); edge1 = intraSubtract(conv0, 32); edge1 = merge([conv0, edge1], mode='concat');
    conv2 = CONV2D(edge1, 64, (3, 3)); edge1 = intraSubtract(conv2, 32); edge1 = merge([conv2, edge1], mode='concat');
    conv0 = interSubtract(conv0, conv2); 
    conv0 = merge([edge1, conv0], mode='concat'); 
    

    conv0 = merge([UpSampling2D(size=(2, 2))(conv0), conv1], mode='concat') # size/2
    conv0 = CONV2D(conv0, 32, (3, 3)); edge1 = intraSubtract(conv0, 16); edge1 = merge([conv0, edge1], mode='concat');
    conv1 = CONV2D(edge1, 32, (3, 3)); edge1 = intraSubtract(conv1, 16); edge1 = merge([conv1, edge1], mode='concat');
    conv0 = interSubtract(conv0, conv1);
    conv0 = merge([edge1, conv0], mode='concat');

    conv0 = CONV2D(conv0, classes, (1, 1), activation='sigmoid')
    model = Model(input=inputs, output=conv0)
    model.summary() 
    return model