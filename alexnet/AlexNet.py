from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

class AlexNet:
  @staticmethod
  def build(width, height, depth, classes, act="elu", reg=0.0005):
    input_shape = (height, width, depth)
    chanDim = -1
    model = Sequential()

    if K.image_data_format() == "channels_first":
      input_shape = (depth, height, width)
      chanDim = 1

    # Block 1: CONV 8x8 * 64 => RELU => POOL 3x3 layer set
    model.add(Conv2D(64, (11, 11), strides=(2, 2), input_shape=input_shape, padding="same", kernel_regularizer=l2(reg)))
    model.add(Activation(act))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
    model.add(Dropout(0.25))

    # Block 2: CONV 5x5 * 192 => RELU => POOL 3x3 layer set
    model.add(Conv2D(128, (7, 7), strides=(2, 2), padding="same", kernel_regularizer=l2(reg)))
    model.add(Activation(act))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.25))

    # Block 3: CONV 3x3 * 384 => RELU => CONV 3x3 * 384 => RELU => CONV 3x3 * 256 =>  RELU
    model.add(Conv2D(192, (3, 3), padding="same", kernel_regularizer=l2(reg)))
    model.add(Activation(act))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(256, (3, 3), padding="same", kernel_regularizer=l2(reg)))
    model.add(Activation(act))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(256, (3, 3), padding="same", kernel_regularizer=l2(reg)))
    model.add(Activation(act))
    model.add(BatchNormalization(axis=chanDim))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.25))

    # Block 4: FC => RELU
    model.add(Flatten())
    model.add(Dense(2304, kernel_regularizer=l2(reg)))
    model.add(Activation(act))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Dropout(0.5))

    # Block 5: FC => RELU
    model.add(Dense(576, kernel_regularizer=l2(reg)))
    model.add(Activation(act))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Dropout(0.5))

    # Block 6: Softmax Classifier
    model.add(Dense(classes, kernel_regularizer=l2(reg)))
    model.add(Activation("softmax"))

    return model