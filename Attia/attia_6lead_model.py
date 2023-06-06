from collections import Counter
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from keras.models import Sequential,Model
from keras.layers import concatenate,Activation,Dropout,Dense,ZeroPadding2D
from keras.layers import Input,add,Conv2D, MaxPooling2D,Flatten,BatchNormalization,LSTM

modelName ='EF_Model.h5'

def ReluBN(i,reluLast=True):
     # See https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md#bn----before-or-after-relu
    if reluLast:
        i = BatchNormalization()(i)
        i = Activation('relu')(i)
    else: 
        i = Activation('relu')(i)
        i = BatchNormalization()(i)
    return i

def ConvPoolBlockNx1(nof,width,i,reluLast=True):
    # Create a width x 1 Conv with Nof filteres, run activations and 2x1 "Max pool decimation"
    i = Conv2D(nof, (width,1), padding='same', kernel_initializer="glorot_normal")(i)
    i = ReluBN(i,reluLast)
    i = MaxPooling2D(pool_size=(2,1))(i)
    return i



def BuildModel(segmentLength=512,padTo=512,n_classes=2,reluLast=True):
    # Build a convolutional neural network from predefiend building box (Conv-BN-relu-pool)

    ecgInp = Input(shape=(segmentLength,2,1))

    if padTo>0 and padTo>segmentLength:
        i = ZeroPadding2D(padding=(int((padTo-segmentLength)/2), 0))(ecgInp)
    else:        
        i=ecgInp

    inputs = ecgInp

    i = ConvPoolBlockNx1(16,5,i,reluLast)
    i = ConvPoolBlockNx1(16,5,i,reluLast)
    i = ConvPoolBlockNx1(32,5,i,reluLast)
    i = MaxPooling2D(pool_size=(2,1))(i) # 2*2 = 4
    i = ConvPoolBlockNx1(32,3,i,reluLast)
    i = ConvPoolBlockNx1(64,3,i,reluLast)
    i = ConvPoolBlockNx1(64,3,i,reluLast)
    i = MaxPooling2D(pool_size=(2,1))(i) # 2*2 = 4
    i = Conv2D(128, (1,2), padding='valid', kernel_initializer="glorot_normal")(i)
    i = ReluBN(i,reluLast)
    
    convFeatures = Flatten()(i)
    i = Dense(64,kernel_initializer="glorot_normal")(convFeatures)
    i = ReluBN(i,reluLast)
    i = Dropout(0.5)(i)
    
    i = Dense(32,kernel_initializer="glorot_normal")(i)
    i = ReluBN(i,reluLast)
    i = Dropout(0.5)(i)
    
    i = Dense(n_classes)(i)
    out = Activation('softmax')(i)

    model = Model(inputs=inputs,outputs=[out])
    model.summary()

    opt0 = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',  optimizer=opt0,  metrics=['accuracy'])

    return model
