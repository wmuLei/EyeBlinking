
import tensorflow as tf
from keras.optimizers import Adam, SGD, RMSprop

from keras import layers
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D, Activation, average
from keras.layers import Add, Subtract, Multiply, Average, Maximum, Minimum, Concatenate,Convolution2D
from keras.layers import merge, BatchNormalization

from keras.engine import Layer
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

class BilinearUpsampling(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):
        super(BilinearUpsampling, self).__init__(**kwargs)

        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0], height, width, input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return K.tf.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0], inputs.shape[2] * self.upsampling[1]), align_corners=True)
        else:
            return K.tf.image.resize_bilinear(inputs, (self.output_size[0], self.output_size[1]), align_corners=True)

    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# with short, medium, and long Subtraction
def TAUnet(shape, classes=1):
    inputs = Input(shape) # [512, 512, 3]
    conv0 = BatchNormalization()(inputs)

    def shortSubtract(x0, x1):
        x2 = Subtract()([x0, x1])
        x2 = Subtract()([Lambda(lambda x: K.abs(x))(x2), x2]);
        x2 = Activation('sigmoid')(x2)
        return x2

    def longSubtract(x0, x1, filter_num=32):
        x0 = AveragePooling2D(pool_size=(2, 2))(x0)
        x0 = CONV2D(x0, filter_num, (1, 1))
        x1 = CONV2D(x1, filter_num, (1, 1))        
        x3 = shortSubtract(x0, x1)
        return x3

    def longSubtract2(x0, x1, filter_num=32):
        x0 = BilinearUpsampling(upsampling=(2, 2))(x0)
        x0 = CONV2D(x0, filter_num, (1, 1))
        x1 = CONV2D(x1, filter_num, (1, 1))
        x3 = shortSubtract(x0, x1)
        return x3

    conv0 = CONV2D(conv0, 32, (3, 3));    
    conv1a = CONV2D(conv0, 32, (3, 3)); 
    edge1 = shortSubtract(conv0, conv1a);
    conv1 = merge([conv1a, edge1], mode='concat', concat_axis=3);    
    conv0 = MaxPooling2D(pool_size=(2, 2))(conv1);  # 512/2

    conv0 = CONV2D(conv0, 64, (3, 3));    
    conv2a = CONV2D(conv0, 64, (3, 3)); 
    edge1 = shortSubtract(conv0, conv2a); 
    edge2 = longSubtract(conv1a, conv2a);
    conv2 = merge([conv2a, edge1, edge2], mode='concat', concat_axis=3);    
    conv0 = MaxPooling2D(pool_size=(2, 2))(conv2);  # 512/4

    conv0 = CONV2D(conv0, 128, (3, 3));    
    conv3a = CONV2D(conv0, 128, (3, 3)); 
    edge1 = shortSubtract(conv0, conv3a);  
    edge2 = longSubtract(conv2a, conv3a);
    conv3 = merge([conv3a, edge1, edge2], mode='concat', concat_axis=3);    
    conv0 = MaxPooling2D(pool_size=(2, 2))(conv3);  # 512/8

    conv0 = CONV2D(conv0, 256, (3, 3));    
    conv4a = CONV2D(conv0, 256, (3, 3)); 
    edge1 = shortSubtract(conv0, conv4a);  
    edge2 = longSubtract(conv3a, conv4a);
    conv4 = merge([conv4a, edge1, edge2], mode='concat', concat_axis=3);    
    conv0 = MaxPooling2D(pool_size=(2, 2))(conv4);  # 512/16

    conv0 = CONV2D(conv0, 512, (3, 3));    
    conv5a = CONV2D(conv0, 512, (3, 3)); 
    edge1 = shortSubtract(conv0, conv5a);  
    edge2 = longSubtract(conv4a, conv5a);
    conv5 = merge([conv5a, edge1, edge2], mode='concat', concat_axis=3);    
    conv0 = MaxPooling2D(pool_size=(2, 2))(conv5);  # 512/32

    #----------------------------------------------
    conv0 = CONV2D(conv0, 1024, (3, 3));    
    conv6a = CONV2D(conv0, 1024, (3, 3)); 
    edge1 = shortSubtract(conv0, conv6a);   
    edge2 = longSubtract(conv5a, conv6a);
    conv0 = merge([conv6a, edge1, edge2], mode='concat', concat_axis=3);  # 512/32
    #----------------------------------------------

    conv0 = merge([UpSampling2D(size=(2, 2))(conv0), conv5], mode='concat', concat_axis=3) # 512/16
    conv0 = CONV2D(conv0, 512, (3, 3));    
    conv7a = CONV2D(conv0, 512, (3, 3)); 
    edge1 = shortSubtract(conv0, conv7a);  
    edge2 = shortSubtract(conv5a, conv7a);  
    edge3 = longSubtract2(conv6a, conv7a);
    conv0 = merge([conv7a, edge1, edge2, edge3], mode='concat', concat_axis=3);


    conv0 = merge([UpSampling2D(size=(2, 2))(conv0), conv4], mode='concat', concat_axis=3) # 512/8
    conv0 = CONV2D(conv0, 256, (3, 3));    
    conv8a = CONV2D(conv0, 256, (3, 3)); 
    edge1 = shortSubtract(conv0, conv8a);  
    edge2 = shortSubtract(conv4a, conv8a);  
    edge3 = longSubtract2(conv7a, conv8a);
    conv0 = merge([conv8a, edge1, edge2, edge3], mode='concat', concat_axis=3);


    conv0 = merge([UpSampling2D(size=(2, 2))(conv0), conv3], mode='concat', concat_axis=3) # 512/4
    conv0 = CONV2D(conv0, 128, (3, 3));    
    conv9a = CONV2D(conv0, 128, (3, 3)); 
    edge1 = shortSubtract(conv0, conv9a);  
    edge2 = shortSubtract(conv3a, conv9a);  
    edge3 = longSubtract2(conv8a, conv9a);
    conv0 = merge([conv9a, edge1, edge2, edge3], mode='concat', concat_axis=3);


    conv0 = merge([UpSampling2D(size=(2, 2))(conv0), conv2], mode='concat', concat_axis=3) # 512/4
    conv0 = CONV2D(conv0, 64, (3, 3));    
    convA = CONV2D(conv0, 64, (3, 3)); 
    edge1 = shortSubtract(conv0, convA);  
    edge2 = shortSubtract(conv2a, convA);    
    edge3 = longSubtract2(conv9a, convA);
    conv0 = merge([convA, edge1, edge2, edge3], mode='concat', concat_axis=3);
    

    conv0 = merge([UpSampling2D(size=(2, 2))(conv0), conv1], mode='concat', concat_axis=3) # 512/2
    conv0 = CONV2D(conv0, 32, (3, 3));    
    convB = CONV2D(conv0, 32, (3, 3)); 
    edge1 = shortSubtract(conv0, convB);  
    edge2 = shortSubtract(conv1a, convB);    
    edge3 = longSubtract2(convA, convB);
    conv0 = merge([convB, edge1, edge2, edge3], mode='concat', concat_axis=3);

    conv0 = CONV2D(conv0, classes, (1, 1), activation='sigmoid')
    model = Model(input=inputs, output=conv0)
    model.summary() 
    return model
