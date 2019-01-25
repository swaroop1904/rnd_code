import argparse
import os
import numpy as np
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D)
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.callbacks import TensorBoard
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.utils import multi_gpu_model

from data_generator_ucf import DataGenerator
from time import time


class decay_lr(Callback):
    '''
        n_epoch = no. of epochs after which the learning rate should be decayed.
        decay = decay value
    '''
    def __init__(self, n_epoch, decay):
        super(decay_lr, self).__init__()
        self.n_epoch=n_epoch
        self.decay=decay

    def on_epoch_begin(self, epoch, logs={}):
        if epoch != 0 and epoch %self.n_epoch == 0:
            K.set_value(self.model.optimizer.lr, K.get_value(self.model.optimizer.lr)*self.decay)

def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\n'.format(
                i, loss[i], acc[i]))

def main():
    parser = argparse.ArgumentParser(
        description='simple 3D convolution for action recognition')
    parser.add_argument('--batch', type=int, default=24)
    parser.add_argument('--epoch', type=int, default=24)
    parser.add_argument('--videos', type=str, default='dataset/train',
                        help='directory where videos are stored')
    parser.add_argument('--nclass', type=int, default=101)
    parser.add_argument('--output', type=str, default='output')
    parser.add_argument('--color', type=bool, default=True)
    parser.add_argument('--skip', type=bool, default=False)
    parser.add_argument('--depth', type=int, default=16)
    args = parser.parse_args()

    # check if you should resize the image
    img_rows, img_cols, frames = 160, 120, args.depth
    channel = 3 if args.color else 1

    model = Sequential()
    model.add(Conv3D(64, kernel_size=(3, 3, 3), input_shape=(img_cols, img_rows,  frames, channel),
                                                            padding='same', activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 1), padding='same'))

    model.add(Conv3D(128, kernel_size=(3, 3, 3), padding='same', activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))

    model.add(Conv3D(256, kernel_size=(3, 3, 3), padding='same', activation='relu'))
    #model.add(Conv3D(256, kernel_size=(3, 3, 3), padding='same', activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))

    model.add(Conv3D(256, kernel_size=(3, 3, 3), padding='same', activation='relu'))
    #model.add(Conv3D(512, kernel_size=(3, 3, 3), padding='same', activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))

    model.add(Conv3D(256, kernel_size=(3, 3, 3), padding='same', activation='relu'))
    #model.add(Conv3D(512, kernel_size=(3, 3, 3), padding='same', activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))

    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(args.nclass, activation='softmax'))

    #model = multi_gpu_model(model, gpus=4)
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(lr=0.00003), metrics=['accuracy'])
    model.summary()
    #model.load_weights('sports1M_weights.h5')

    filepath="d_3dcnnmodel_ucf_lr1_decay-{epoch:02d}-{val_acc:.2f}.hd5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()), write_graph=True )
    decaySchedule=decay_lr(4, 0.10)
    callbacks_list = [tensorboard, decaySchedule, checkpoint]

    train_generator = DataGenerator(args.batch, 'UCF101/train', frames, True, False)

    test_generator = DataGenerator(args.batch, 'UCF101/test', frames, False, False)

    validation_generator = DataGenerator(args.batch, 'UCF101/validation', frames, False, True)

    history = model.fit_generator(generator=train_generator,
                         epochs=args.epoch, shuffle=False, validation_data=validation_generator, callbacks=callbacks_list)
                         #use_multiprocessing=True, workers=6)

    loss, acc = model.evaluate_generator(test_generator)

    model_json = model.to_json()
    with open('output/action_3dcnnmodel-gpu_ucf_lr1_decay.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('output/action_3dcnnmodel-gpu_ucf_lr1_decay.hd5')

    print('Test loss:', loss)
    print('Test accuracy:', acc)

    save_history(history, args.output)

if __name__ == '__main__':
    main()
