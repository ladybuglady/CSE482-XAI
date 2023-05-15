from keras.metrics import categorical_crossentropy
import tensorflow as tf
from keras.layers import Dense,Activation,Dropout
from keras.layers import LSTM,Bidirectional, GRU #could try TimeDistributed(Dense(...))
from keras.models import Sequential, load_model
from keras import optimizers,regularizers
from keras.layers import BatchNormalization
import keras.backend as K
from keras.metrics import categorical_accuracy
from keras.callbacks import EarlyStopping
from keras import regularizers

modelName ='EF_Model.h5'

def lstm_model():
  segmentLength=5000

  nadam_opt = tf.keras.optimizers.Nadam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    weight_decay=None,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    jit_compile=True,
    name="Nadam")
  
  focal_loss = tf.keras.losses.BinaryFocalCrossentropy(
    apply_class_balancing=False,
    alpha=0.25,
    gamma=2.0,
    from_logits=False,
    label_smoothing=0.0,
    axis=-1,
    name='binary_focal_crossentropy')
  
  model = Sequential()
  model.add(LSTM(64, input_shape=(segmentLength, 2)))
  model.add(Dense(32))
  model.add(Dense(2, activation='softmax'))
  model.compile(loss=focal_loss, optimizer=nadam_opt, metrics=['accuracy'])
  print(model.summary())
  return(model)

# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6815557/