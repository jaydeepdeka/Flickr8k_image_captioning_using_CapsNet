# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 12:19:11 2019

@author: jayde
"""
# Import statements
from keras import layers, models
from keras.layers import Input, Conv2D, Activation, Dense, Reshape
from keras.layers import BatchNormalization, Deconvolution2D
from keras.applications.vgg16 import VGG16

# Import the Capsule Layer classes
from utils.capslayers import Conv2DCaps
from utils.capslayers import ConvCapsuleLayer3D
from utils.capslayers import CapsuleLayer
from utils.capslayers import CapsToScalars
from utils.capslayers import Mask_CID
from utils.capslayers import ConvertToCaps
from utils.capslayers import FlattenCaps


def load_DeepCapsNet(input_shape, n_class, routings, weights):
    x = Input(shape=input_shape)
    l = x

    l = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding="same")(l)  # common conv layer
    l = BatchNormalization()(l)
    l = ConvertToCaps()(l)

    l = Conv2DCaps(32, 4, kernel_size=(3, 3), strides=(2, 2), r_num=1, b_alphas=[1, 1, 1])(l)
    l_skip = Conv2DCaps(32, 4, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = Conv2DCaps(32, 4, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = Conv2DCaps(32, 4, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = layers.Add()([l, l_skip])

    l = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(2, 2), r_num=1, b_alphas=[1, 1, 1])(l)
    l_skip = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = layers.Add()([l, l_skip])

    l = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(2, 2), r_num=1, b_alphas=[1, 1, 1])(l)
    l_skip = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = layers.Add()([l, l_skip])
    l1 = l

    l = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(2, 2), r_num=1, b_alphas=[1, 1, 1])(l)
    l_skip = ConvCapsuleLayer3D(kernel_size=3, num_capsule=32, num_atoms=8, strides=1, padding='same', routings=3)(l)
    l = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = layers.Add()([l, l_skip])
    l2 = l

    la = FlattenCaps()(l2)
    lb = FlattenCaps()(l1)
    l = layers.Concatenate(axis=-2)([la, lb])
    digits_caps = CapsuleLayer(num_capsule=n_class, dim_capsule=32, routings=routings, channels=0, name='digit_caps')(l)

    l = CapsToScalars(name='capsnet')(digits_caps)

    m_capsnet = models.Model(inputs=x, outputs=l, name='capsnet_model')

    y = Input(shape=(n_class,))

    masked_by_y = Mask_CID()([digits_caps, y])  
    masked = Mask_CID()(digits_caps)

    # Decoder Network
    decoder = models.Sequential(name='decoder')
    decoder.add(Dense(input_dim=32, activation="relu", output_dim=8 * 8 * 16))
    decoder.add(Reshape((8, 8, 16)))
    decoder.add(BatchNormalization(momentum=0.8))
    decoder.add(Deconvolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
    decoder.add(Deconvolution2D(32, 3, 3, subsample=(2, 2), border_mode='same'))
    decoder.add(Deconvolution2D(16, 3, 3, subsample=(2, 2), border_mode='same'))
    decoder.add(Deconvolution2D(8, 3, 3, subsample=(2, 2), border_mode='same'))
    decoder.add(Deconvolution2D(3, 3, 3, subsample=(1, 1), border_mode='same'))
    decoder.add(Activation("relu"))
    decoder.add(Reshape(target_shape=(64, 64, 3), name='out_recon'))

    train_model = models.Model([x, y], [m_capsnet.output, decoder(masked_by_y)])
    eval_model = models.Model(x, [m_capsnet.output, decoder(masked)])
    print("Complete Capsule Architecture")
    train_model.summary()
    eval_model.load_weights(weights)
    # Popping out the decoder layers and ouput layer
    for _ in range(3):
        eval_model.layers.pop()
    print("Capsule Network as feature extractor")
    eval_model.summary()

    return eval_model

def load_VGG():
    model = VGG16()
    print("Complete Capsule Architecture")
    model.summary()
	# re-structure the model
    model.layers.pop()
    print("VGG16 as feature extractor")
    model.summary()
    return model