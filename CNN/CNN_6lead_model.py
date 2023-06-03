from collections import Counter
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from keras.models import Sequential,Model
from keras.layers import concatenate,Activation,Dropout,Dense,ZeroPadding2D
from keras.layers import Input,add,Conv2D, MaxPooling2D,Flatten,BatchNormalization,LSTM

modelName ='EF_Model.h5'


class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,kernel_size=3,padding=1,bias=False):
        super(ResidualBlock,self).__init__()
        self.cnn1 =nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size,1,padding,bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()
            
    def forward(self,x):
        residual = x
        x = self.cnn1(x)
        x = self.cnn2(x)
        x += self.shortcut(residual)
        x = nn.ReLU(True)(x)
        return x
    
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18,self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=2,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        
        self.block2 = nn.Sequential(
            nn.MaxPool2d(1,1),
            ResidualBlock(64,64),
            ResidualBlock(64,64,2)
        )
        
        self.block3 = nn.Sequential(
            ResidualBlock(64,128),
            ResidualBlock(128,128,2)
        )
        
        self.block4 = nn.Sequential(
            ResidualBlock(128,256),
            ResidualBlock(256,256,2)
        )
        self.block5 = nn.Sequential(
            ResidualBlock(256,512),
            ResidualBlock(512,512,2)
        )
        
        self.avgpool = nn.AvgPool2d(2)

        self.fc3 = nn.Linear(512,7)
        self.fc4 = nn.Linear(7, 1)

        self.m = nn.Sigmoid()
        
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc3(x)
        x = self.fc4(x)
        out = self.m(x)
        return out
    
def Block1(i):
    # # filters, kernel size, 
    Conv2D(nof, (width,1), padding='same', kernel_initializer="glorot_normal")(i)
    
    # in channels, out channels, 
    i = nn.Conv2d(1,64,kernel_size=2,stride=2,padding=3,bias=False),

def BuildModel(segmentLength=512,padTo=512,n_classes=2,reluLast=True):

    # Build a convolutional neural network from predefiend building box (Conv-BN-relu-pool)

    ecgInp = Input(shape=(segmentLength,2,1))

    if padTo>0 and padTo>segmentLength:
        i = ZeroPadding2D(padding=(int((padTo-segmentLength)/2), 0))(ecgInp)
    else:        
        i=ecgInp

    inputs = ecgInp

    i = 