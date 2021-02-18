import os
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten, Dropout
from keras.layers.core import Dense
from keras.datasets import cifar10
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import numpy as np
from sakamiti_dataset import SakamitiDataset

class FineTuningInceptionV3():
    """
    Inception-v3ネットワークを利用してfine tuningを行う
    入力層は、Inception-v3の仕様から (n, 139, 139, 3) とする
    出力層は、坂道グループ数 3 とする
    """
    def __init__(self):
        self._num_classes = 3
        pass
    
    if __name__ == "__main__":
        # Inception-v3の重みをダウンロードする
        base_model = InceptionV3(weights="imagenet", include_top=False)
        # 最終層無しのmodelの構造を出力
        print("--------------imported model--------------")
        base_model.summary()
        
        x = base_model.output
        
        x = GlobalAveragePooling2D()(x)
        
        predictions = Dense(3, activation="softmax")(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        print("--------------created model--------------")
        model.summary()
        
        # base model (Inception-v3)は学習させないため、layer.trainable = False に設定する
        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer="rmsprop", loss="categorical_crossentropy",metrics=["accuracy"])
        
        dataset = SakamitiDataset()
        x_train, x_test, y_train, y_test = dataset.get_batch(test_per = 0.1)
        
        print(f'train num is {len(x_train)}')
        
        datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False)
        
        validation_split = 0.2
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        validation_size = int(x_train.shape[0] * validation_split)
        x_train, x_valid = x_train[indices[:-validation_size], :], x_train[indices[-validation_size:], :]
        y_train, y_valid = y_train[indices[:-validation_size], :], y_train[indices[-validation_size:], :]
        
        model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=128),
        steps_per_epoch=x_train.shape[0], # batch size
        epochs=30,
        validation_data=(x_valid,y_valid),
        verbose=1
        )
        
        for layer in model.layers[:249]:
            layer.trainable = False
            
        for layer in model.layers[249:]:
            layer.trainable = True

        from keras.optimizers import SGD
        
        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=["accuracy"])
        
        model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=128),
        steps_per_epoch=x_train.shape[0], # batch size
        epochs=30,
        validation_data=(x_valid,y_valid),
        verbose=1
        )
        
        model.save('./model/fine-tuning-sakamiti.h5')
        
        #new_model = keras.models.load_model('path_to_my_model.h5')