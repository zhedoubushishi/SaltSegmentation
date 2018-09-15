# encoding=utf8
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dropout, BatchNormalization, Activation, Add
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
from src.config import *
from src.utils import my_iou_metric, my_iou_metric_2

################################################################################
# train u-net & resnet model with Keras
################################################################################


def batch_activate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def convolution_block(x, filters, size, strides=(1, 1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation:
        x = batch_activate(x)
    return x


def residual_block(block_input, num_filters=16, is_batch_activate=False):
    x = batch_activate(block_input)
    x = convolution_block(x, num_filters, (3, 3))
    x = convolution_block(x, num_filters, (3, 3), activation=False)
    x = Add()([x, block_input])
    if is_batch_activate:
        x = batch_activate(x)
    return x


# Build model
def build_model(input_layer, start_neurons, dropout_ratio=0.5):
    # 101 -> 50
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(input_layer)
    conv1 = residual_block(conv1, start_neurons * 1)
    conv1 = residual_block(conv1, start_neurons * 1, True)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(dropout_ratio / 2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2, start_neurons * 2)
    conv2 = residual_block(conv2, start_neurons * 2, True)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(dropout_ratio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3, start_neurons * 4)
    conv3 = residual_block(conv3, start_neurons * 4, True)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(dropout_ratio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool3)
    conv4 = residual_block(conv4, start_neurons * 8)
    conv4 = residual_block(conv4, start_neurons * 8, True)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(dropout_ratio)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm, start_neurons * 16)
    convm = residual_block(convm, start_neurons * 16, True)

    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(dropout_ratio)(uconv4)

    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4, start_neurons * 8)
    uconv4 = residual_block(uconv4, start_neurons * 8, True)

    # 12 -> 25
    # deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(dropout_ratio)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, start_neurons * 4)
    uconv3 = residual_block(uconv3, start_neurons * 4, True)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])

    uconv2 = Dropout(dropout_ratio)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, start_neurons * 2)
    uconv2 = residual_block(uconv2, start_neurons * 2, True)

    # 50 -> 101
    # deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
    uconv1 = concatenate([deconv1, conv1])

    uconv1 = Dropout(dropout_ratio)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, start_neurons * 1)
    uconv1 = residual_block(uconv1, start_neurons * 1, True)

    # uconv1 = Dropout(DropoutRatio/2)(uconv1)
    # output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    output_layer_noActi = Conv2D(1, (1, 1), padding="same", activation=None)(uconv1)
    output_layer = Activation('sigmoid')(output_layer_noActi)

    return output_layer


class KerasModel:
    def __init__(self, x_train, y_train, x_valid, y_valid):
        self.x_train = x_train
        self.x_valid = x_valid
        self.y_train = y_train
        self.y_valid = y_valid

    def train(self, img_size_target):
        # model1
        input_layer = Input((img_size_target, img_size_target, 1))
        output_layer = build_model(input_layer, START_NEURONS, DROPOUT_RATIO)

        model1 = Model(input_layer, output_layer)

        c = optimizers.adam(lr=MODEL1_ADAM_LR)
        model1.compile(loss=MODEL1_LOSS, optimizer=c, metrics=[my_iou_metric])

        # early_stopping = EarlyStopping(monitor='my_iou_metric', mode='max', patience=10, verbose=1)
        model_checkpoint = ModelCheckpoint(SAVE_MODEL_NAME, monitor='my_iou_metric',
                                           mode='max', save_best_only=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='my_iou_metric', mode='max',
                                      factor=0.5, patience=5, min_lr=0.0001, verbose=1)

        history = model1.fit(self.x_train, self.y_train,
                             validation_data=[self.x_valid, self.y_valid],
                             epochs=MODEL1_EPOCHS,
                             batch_size=MODEL1_BATCH_SIZE,
                             callbacks=[model_checkpoint, reduce_lr],
                             verbose=2)

        # model2
        model1 = load_model(SAVE_MODEL_NAME, custom_objects={'my_iou_metric': my_iou_metric})
        # remove layter activation layer and use losvasz loss
        input_x = model1.layers[0].input

        output_layer = model1.layers[-1].input
        model = Model(input_x, output_layer)
        c = optimizers.adam(lr=MODEL2_ADAM_LR)

        # lovasz_loss need input range (-∞，+∞), so cancel the last "sigmoid" activation
        # Then the default threshod for pixel prediction is 0 instead of 0.5, as in my_iou_metric_2.
        model.compile(loss=MODEL2_LOSS, optimizer=c, metrics=[my_iou_metric_2])

        # model.summary()
        early_stopping = EarlyStopping(monitor='val_my_iou_metric_2', mode='max', patience=20, verbose=1)
        model_checkpoint = ModelCheckpoint(SAVE_MODEL_NAME, monitor='val_my_iou_metric_2',
                                           mode='max', save_best_only=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric_2',
                                      mode='max', factor=0.5, patience=5, min_lr=0.0001, verbose=1)

        history = model.fit(self.x_train, self.y_train,
                            validation_data=[self.x_valid, self.y_valid],
                            epochs=MODEL2_EPOCHS,
                            batch_size=MODEL2_BATCH_SIZE,
                            callbacks=[model_checkpoint, reduce_lr, early_stopping],
                            verbose=2)

    @staticmethod
    def predict(x_test, img_size_target):
        model = load_model(SAVE_MODEL_NAME, custom_objects={'my_iou_metric_2': my_iou_metric_2,
                                                            'lovasz_loss': lovasz_loss})
        x_test_reflect = np.array([np.fliplr(x) for x in x_test])
        preds_test = model.predict(x_test).reshape(-1, img_size_target, img_size_target)
        preds_test2_refect = model.predict(x_test_reflect).reshape(-1, img_size_target, img_size_target)
        preds_test += np.array([np.fliplr(x) for x in preds_test2_refect])
        return preds_test / 2
