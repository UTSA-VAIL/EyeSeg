import os
import tensorflow as tf
import tensorflow.keras.backend as K 
from tensorflow.keras import Model, Input
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import LeakyReLU, PReLU, Conv2D, Dropout, \
    UpSampling2D, ReLU, Reshape, GlobalAveragePooling2D, AveragePooling2D, \
    BatchNormalization, concatenate, SpatialDropout2D, Activation, Conv2DTranspose, add
class input_layer:
    '''
    '''
    def __call__(input_shape = None):
        assert input_shape is not None
        return Input(input_shape)

class initial_block:

    def __init__(self, args, padding = 'same', kernel_size = (3,3)):
        '''
        '''
        # Set Dilation rate
        dilation_rate1 = 2 if args.DILATED else 1
        dilation_rate2 = 4 if args.DILATED else 1

        # Convolutional Layers
        self.conv1 = Conv2D(args.FILTERS, kernel_size=kernel_size, padding = padding, input_shape=args.INPUT_SHAPE)

        # Activation Layer
        self.activation1 = LeakyReLU()

        # Dropout layer
        #self.dropout = Dropout(args.DROPOUT_RATE)

        # Batch Normalizer
        self.bn = BatchNormalization(momentum=0.1)

    
    def __call__(self, model, training):
        '''
        '''
        layer1 = self.conv1(model)
        layer1 = self.activation1(layer1)
        out = self.bn(layer1)
        #out = self.dropout(out) if training else out
        return out
        


class encoder_block:

    def __init__(self, args, training, padding = 'same', kernel_size = (3,3)):
        '''
        '''
        # Set Dilation rate
        dilation_rate1 = 2 if args.DILATED else 1
        dilation_rate2 = 4 if args.DILATED else 1

        # Convolutional Layers
        self.conv1 = Conv2D(args.FILTERS, kernel_size=kernel_size, padding = padding)
        self.conv2 = Conv2D(args.FILTERS, kernel_size=kernel_size, padding = padding, dilation_rate=dilation_rate1)
        self.conv3 = Conv2D(args.FILTERS, kernel_size=kernel_size, padding = padding)
        self.conv4 = Conv2D(args.FILTERS, kernel_size=kernel_size, padding = padding)
        self.conv5 = Conv2D(args.FILTERS, kernel_size=kernel_size, padding = padding)
        #self.conv8 = Conv2D(args.FILTERS, kernel_size=kernel_size, padding = padding)

        # Activation Layers
        self.activation1 = PReLU(shared_axes=[1,2])
        self.activation2 = PReLU(shared_axes=[1,2])
        self.activation3 = PReLU(shared_axes=[1,2])

        # Dropout layer
        self.dropout1 = Dropout(args.DROPOUT_RATE)
        self.dropout2 = Dropout(args.DROPOUT_RATE)
        self.dropout3 = Dropout(args.DROPOUT_RATE)

        # Downsampling
        self.down_sample = AveragePooling2D((2,2))
        self.localization = GlobalAveragePooling2D()

        # Batch Normalizer
        self.bn = BatchNormalization(momentum=0.1)
    
    def __call__(self, input_block, training, down = True):
        '''
        '''
        initial = self.down_sample(input_block) if down else input_block
        layer1 = self.conv1(initial)
        layer1 = self.dropout1(layer1)
        layer1 = self.activation1(layer1)
        layer2_cat = concatenate([initial, layer1])
        layer2 = self.conv2(layer2_cat)
        layer2 = self.conv3(layer2)
        layer2 = self.dropout2(layer2)
        layer2 = self.activation2(layer2)
        layer3 = concatenate([layer2_cat, layer2])
        layer3 = self.conv4(layer3)
        layer3 = self.conv5(layer3)
        layer3 = self.dropout3(layer3)
        layer3 = self.activation3(layer3)
        return self.bn(layer3)


class decoder_block:

    def __init__(self, args, training, padding = 'same', kernel_size = (3,3), previous_features = 32):
        '''
        '''
        # Convolutional Layers
        self.conv1 = Conv2D(args.FILTERS + previous_features, kernel_size=kernel_size, padding = padding)
        self.conv2 = Conv2D(args.FILTERS, kernel_size=kernel_size, padding = padding)
        self.conv3 = Conv2D(args.FILTERS, kernel_size=kernel_size, padding = padding)
        self.conv4 = Conv2D(args.FILTERS, kernel_size=kernel_size, padding = padding)

        # Activation Layers
        self.activation1 = LeakyReLU()
        self.activation2 = LeakyReLU()


        # Dropout layer
        self.dropout1 = Dropout(args.DROPOUT_RATE)
        self.dropout2 = Dropout(args.DROPOUT_RATE)

        # Upsampling
        self.up_sample = UpSampling2D()
    
    def __call__(self, model, skip_connection, training, upsample = True):
        '''
        '''
        up = self.up_sample(model)
        up_cat = concatenate([up, skip_connection])
        
        layer1 = self.conv1(up_cat)
        layer1 = self.conv2(layer1)
        layer1 = self.dropout1(layer1)
        layer1 = self.activation1(layer1)
        layer2 = concatenate([up_cat, layer1])
        layer2 = self.conv3(layer2)
        layer2 = self.conv4(layer2)
        layer2 = self.dropout2(layer2)
        return self.activation2(layer2)
        
class EyeSeg:
    
    def __init__(self, args):
        '''

        '''
        self.FILTER_SIZE = args.FILTER_SIZE
        self.NUM_CLASSES = args.NUM_CLASSES
        self.DROPOUT = args.DROPOUT
        self.INPUT_SHAPE = args.INPUT_SHAPE
        self.OUTPUT_SHAPE = args.OUTPUT_SHAPE
        self.TRAIN = args.TRAIN
        self.input_block = initial_block(args)
        self.encoder_blocks = {block+1:encoder_block(args, self.TRAIN) for block in range(5)}
        self.decoder_blocks = {block+1:decoder_block(args, self.TRAIN) for block in range(4)}

    def __call__(self, inputs, training = False):
        '''
        '''
        encoder_block_1 = self.encoder_blocks[1](inputs, training, down=False)
        encoder_block_2 = self.encoder_blocks[2](encoder_block_1, training)
        encoder_block_3 = self.encoder_blocks[3](encoder_block_2, training)
        encoder_block_4 = self.encoder_blocks[4](encoder_block_3, training)
        
        decoder_block_5 = self.decoder_blocks[3](encoder_block_4,encoder_block_3, training)
        decoder_block_6 = self.decoder_blocks[4](decoder_block_5,encoder_block_2, training)
        decoder_block_7 = self.decoder_blocks[1](decoder_block_6,encoder_block_1, training)
        out = Conv2DTranspose(self.NUM_CLASSES, (1,1))(decoder_block_7)
        return Activation(softmax)(out)



if __name__ == '__main__':
    print('CWD:',os.getcwd())