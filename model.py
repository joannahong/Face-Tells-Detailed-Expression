# -*- coding: utf-8 -*-
## 마지막 단계

import tensorflow as tf

from keras import backend as K

from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Convolution2D
from keras.layers import Convolution1D
from keras.layers import ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Reshape
from keras.layers import Dense
from keras.layers import Input
from keras.layers import concatenate
from keras.layers import Conv2D
from keras.layers import UpSampling2D

from keras.layers import add
from keras.layers import BatchNormalization


import numpy as np
import math

from base_model import BaseModel

class CaptionGenerator(BaseModel):
    def build(self):
        """ Build the model. """
        self.build_cnn()
        self.build_rnn()
        if self.is_train:
            self.build_optimizer()
            self.build_summary()

    def build_cnn(self):
        """ Build the CNN. """
        print("Building the CNN...")
        if self.config.cnn == 'vgg16':
            self.build_vgg16()
        if self.config.cnn == 'vggface':
            self.build_vggface()
        elif self.config.cnn == 'vggface_emotion':
            self.build_vggface_emotion()
        elif self.config.cnn == 'vgg16_modified':
            self.build_vgg16_modified()
        else:
            self.build_resnet50()
        print("CNN built.")

    def build_vgg16(self):
        """ Build the VGG16 net. """
        config = self.config

        images = tf.placeholder(
            dtype = tf.float32,
            shape = [config.batch_size] + self.image_shape)

        conv1_1_feats = self.nn.conv2d(images, 64, name = 'conv1_1')
        conv1_2_feats = self.nn.conv2d(conv1_1_feats, 64, name = 'conv1_2')
        pool1_feats = self.nn.max_pool2d(conv1_2_feats, name = 'pool1')

        conv2_1_feats = self.nn.conv2d(pool1_feats, 128, name = 'conv2_1')
        conv2_2_feats = self.nn.conv2d(conv2_1_feats, 128, name = 'conv2_2')
        pool2_feats = self.nn.max_pool2d(conv2_2_feats, name = 'pool2')

        conv3_1_feats = self.nn.conv2d(pool2_feats, 256, name = 'conv3_1')
        conv3_2_feats = self.nn.conv2d(conv3_1_feats, 256, name = 'conv3_2')
        conv3_3_feats = self.nn.conv2d(conv3_2_feats, 256, name = 'conv3_3')
        pool3_feats = self.nn.max_pool2d(conv3_3_feats, name = 'pool3')

        conv4_1_feats = self.nn.conv2d(pool3_feats, 512, name = 'conv4_1')
        conv4_2_feats = self.nn.conv2d(conv4_1_feats, 512, name = 'conv4_2')
        conv4_3_feats = self.nn.conv2d(conv4_2_feats, 512, name = 'conv4_3')
        pool4_feats = self.nn.max_pool2d(conv4_3_feats, name = 'pool4')

        conv5_1_feats = self.nn.conv2d(pool4_feats, 512, name = 'conv5_1')
        conv5_2_feats = self.nn.conv2d(conv5_1_feats, 512, name = 'conv5_2')
        conv5_3_feats = self.nn.conv2d(conv5_2_feats, 512, name = 'conv5_3')

        reshaped_conv5_3_feats = tf.reshape(conv5_3_feats,
                                            [config.batch_size, 196, 512])

        self.conv_feats = reshaped_conv5_3_feats
        self.num_ctx = 196
        self.dim_ctx = 512
        self.images = images

    def build_vgg16_modified(self):
        """ Build the VGG16 net. """
        config = self.config

        images = tf.placeholder(
            dtype = tf.float32,
            shape = [config.batch_size] + self.image_shape)

        conv1_1_feats = self.nn.conv2d(images, 32, name = 'conv1_1')
        conv1_2_feats = self.nn.conv2d(conv1_1_feats, 32, name = 'conv1_2')
        pool1_feats = self.nn.max_pool2d(conv1_2_feats, name = 'pool1')

        conv2_1_feats = self.nn.conv2d(pool1_feats, 64, name = 'conv2_1')
        conv2_2_feats = self.nn.conv2d(conv2_1_feats, 64, name = 'conv2_2')
        pool2_feats = self.nn.max_pool2d(conv2_2_feats, name = 'pool2')

        conv3_1_feats = self.nn.conv2d(pool2_feats, 128, name = 'conv3_1')
        conv3_2_feats = self.nn.conv2d(conv3_1_feats, 128, name = 'conv3_2')
        conv3_3_feats = self.nn.conv2d(conv3_2_feats, 128, name = 'conv3_3')
        pool3_feats = self.nn.max_pool2d(conv3_3_feats, name = 'pool3')

        conv4_1_feats = self.nn.conv2d(pool3_feats, 256, name = 'conv4_1')
        conv4_2_feats = self.nn.conv2d(conv4_1_feats, 256, name = 'conv4_2')
        conv4_3_feats = self.nn.conv2d(conv4_2_feats, 256, name = 'conv4_3')
        pool4_feats = self.nn.max_pool2d(conv4_3_feats, name = 'pool4')

        conv5_1_feats = self.nn.conv2d(pool4_feats, 512, name = 'conv5_1')
        conv5_2_feats = self.nn.conv2d(conv5_1_feats, 512, name = 'conv5_2')
        conv5_3_feats = self.nn.conv2d(conv5_2_feats, 256, name = 'conv5_3')

        ## Added part
        pooltemp1_1 = self.nn.max_pool2d(conv1_2_feats, name = 'pool1_1')
        pooltemp1_2 = self.nn.max_pool2d(pooltemp1_1, name = 'pool1_2')
        pooltemp1_3 = self.nn.max_pool2d(pooltemp1_2, name = 'pool1_3')
        pooltemp1_4 = self.nn.max_pool2d(pooltemp1_3, name = 'pool1_4')
        pooltemp1_5 = self.nn.conv2d(pooltemp1_4, 256, name = 'pool1_5')

        pooltemp2_1 = self.nn.max_pool2d(conv2_2_feats, name = 'pool2_1')
        pooltemp2_2 = self.nn.max_pool2d(pooltemp2_1, name = 'pool2_2')
        pooltemp2_3 = self.nn.max_pool2d(pooltemp2_2, name = 'pool2_3')
        pooltemp2_4 = self.nn.conv2d(pooltemp2_3, 256, name = 'pool2_5')

        pooltemp3_1 = self.nn.max_pool2d(conv3_3_feats, name = 'pool3_1')
        pooltemp3_2 = self.nn.max_pool2d(pooltemp3_1, name = 'pool3_2')
        pooltemp3_3 = self.nn.conv2d(pooltemp3_2, 256, name = 'pool3_3')

        pooltemp4_1 = self.nn.conv2d(pool4_feats, 256, name = 'pool4_1')

        skip1 = tf.concat([pooltemp1_5, pooltemp2_4], 3)
        skip2 = tf.concat([skip1,pooltemp3_3], 3)
        skip3 = tf.concat([skip2,pooltemp4_1], 3)

        reshaped_conv5_3_feats = tf.reshape(skip3,
                                            [config.batch_size, 14*14, 1024])

        self.conv_feats = reshaped_conv5_3_feats
        self.num_ctx = 14*14
        self.dim_ctx = 1024
        self.images = images

    def build_vggface(self):
        # keras implementation
        # source from: https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/
        config = self.config
        K.set_learning_phase(1)

        input_feats = Input(shape=self.image_shape)
        self.images = input_feats
        input_feats = _augment(input_feats, resize=[224, 224])
        aug_feats = Input(shape=self.image_shape, tensor=input_feats)
        zeropad1_1_feats = ZeroPadding2D((1, 1), input_shape=(224, 224, 3))(aug_feats)
        conv1_1_feats = Convolution2D(64, (3, 3), activation='relu')(zeropad1_1_feats)
        zeropad1_2_feats = ZeroPadding2D((1, 1))(conv1_1_feats)
        conv1_2_feats = Convolution2D(64, (3, 3), activation='relu')(zeropad1_2_feats)
        pool1_feats = (MaxPooling2D((2, 2), strides=(2, 2)))(conv1_2_feats)

        zeropad2_1_feats = ZeroPadding2D((1, 1), input_shape=(224, 224, 3))(pool1_feats)
        conv2_1_feats = Convolution2D(128, (3, 3), activation='relu')(zeropad2_1_feats)
        zeropad2_2_feats = ZeroPadding2D((1, 1))(conv2_1_feats)
        conv2_2_feats = Convolution2D(128, (3, 3), activation='relu')(zeropad2_2_feats)
        pool2_feats = (MaxPooling2D((2, 2), strides=(2, 2)))(conv2_2_feats)

        zeropad3_1_feats = ZeroPadding2D((1, 1), input_shape=(224, 224, 3))(pool2_feats)
        conv3_1_feats = Convolution2D(256, (3, 3), activation='relu')(zeropad3_1_feats)
        zeropad3_2_feats = ZeroPadding2D((1, 1))(conv3_1_feats)
        conv3_2_feats = Convolution2D(256, (3, 3), activation='relu')(zeropad3_2_feats)
        zeropad3_3_feats = ZeroPadding2D((1, 1))(conv3_2_feats)
        conv3_3_feats = Convolution2D(256, (3, 3), activation='relu')(zeropad3_3_feats)
        pool3_feats = (MaxPooling2D((2, 2), strides=(2, 2)))(conv3_3_feats)

        zeropad4_1_feats = ZeroPadding2D((1, 1), input_shape=(224, 224, 3))(pool3_feats)
        conv4_1_feats = Convolution2D(512, (3, 3), activation='relu')(zeropad4_1_feats)
        zeropad4_2_feats = ZeroPadding2D((1, 1))(conv4_1_feats)
        conv4_2_feats = Convolution2D(512, (3, 3), activation='relu')(zeropad4_2_feats)
        zeropad4_3_feats = ZeroPadding2D((1, 1))(conv4_2_feats)
        conv4_3_feats = Convolution2D(512, (3, 3), activation='relu')(zeropad4_3_feats)
        pool4_feats = (MaxPooling2D((2, 2), strides=(2, 2)))(conv4_3_feats)

        zeropad5_1_feats = ZeroPadding2D((1, 1), input_shape=(224, 224, 3))(pool4_feats)
        conv5_1_feats = Convolution2D(512, (3, 3), activation='relu')(zeropad5_1_feats)
        zeropad5_2_feats = ZeroPadding2D((1, 1))(conv5_1_feats)
        conv5_2_feats = Convolution2D(512, (3, 3), activation='relu')(zeropad5_2_feats)
        zeropad5_3_feats = ZeroPadding2D((1, 1))(conv5_2_feats)
        conv5_3_feats = Convolution2D(512, (3, 3), activation='relu')(zeropad5_3_feats)
        pool5_feats = (MaxPooling2D((2, 2), strides=(2, 2)))(conv5_3_feats)

        fully1_feats = (Convolution2D(4096, (7, 7), activation='relu'))(pool5_feats)
        dropout1_feats = Dropout(0.5)(fully1_feats)
        fully2_feats = (Convolution2D(4096, (1, 1), activation='relu'))(dropout1_feats)
        dropout1_feats = Dropout(0.5)(fully2_feats)
        fully3_feats = (Convolution2D(2622, (1, 1), activation='relu'))(dropout1_feats)
        dropout1_feats = Flatten()(fully3_feats)
        softmax_feats = Activation('softmax')(dropout1_feats)

        model = Model(input_feats, softmax_feats)
        model.load_weights('vgg_face_weights.h5')

        for layer in model.layers:
            layer.trainable = False

        self.conv_feats = Reshape((196, 512))(conv5_3_feats)
        self.num_ctx = 196
        self.dim_ctx = 512

    def build_vggface_emotion(self):
        # keras implementation
        # source from: https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/
        config = self.config
        K.set_learning_phase(1)

        input_feats = Input(shape=self.image_shape)
        self.images = input_feats
        input_feats = _augment(input_feats, resize=[224, 224])
        aug_feats = Input(shape=self.image_shape, tensor=input_feats)
        zeropad1_1_feats = ZeroPadding2D((1, 1), input_shape=(224, 224, 3))(aug_feats)
        conv1_1_feats = Convolution2D(64, (3, 3), activation='relu')(zeropad1_1_feats)
        zeropad1_2_feats = ZeroPadding2D((1, 1))(conv1_1_feats)
        conv1_2_feats = Convolution2D(64, (3, 3), activation='relu')(zeropad1_2_feats)
        pool1_feats = (MaxPooling2D((2, 2), strides=(2, 2)))(conv1_2_feats)

        zeropad2_1_feats = ZeroPadding2D((1, 1), input_shape=(224, 224, 3))(pool1_feats)
        conv2_1_feats = Convolution2D(128, (3, 3), activation='relu')(zeropad2_1_feats)
        zeropad2_2_feats = ZeroPadding2D((1, 1))(conv2_1_feats)
        conv2_2_feats = Convolution2D(128, (3, 3), activation='relu')(zeropad2_2_feats)
        pool2_feats = (MaxPooling2D((2, 2), strides=(2, 2)))(conv2_2_feats)

        zeropad3_1_feats = ZeroPadding2D((1, 1), input_shape=(224, 224, 3))(pool2_feats)
        conv3_1_feats = Convolution2D(256, (3, 3), activation='relu')(zeropad3_1_feats)
        zeropad3_2_feats = ZeroPadding2D((1, 1))(conv3_1_feats)
        conv3_2_feats = Convolution2D(256, (3, 3), activation='relu')(zeropad3_2_feats)
        zeropad3_3_feats = ZeroPadding2D((1, 1))(conv3_2_feats)
        conv3_3_feats = Convolution2D(256, (3, 3), activation='relu')(zeropad3_3_feats)
        pool3_feats = (MaxPooling2D((2, 2), strides=(2, 2)))(conv3_3_feats)

        zeropad4_1_feats = ZeroPadding2D((1, 1), input_shape=(224, 224, 3))(pool3_feats)
        conv4_1_feats = Convolution2D(512, (3, 3), activation='relu')(zeropad4_1_feats)
        zeropad4_2_feats = ZeroPadding2D((1, 1))(conv4_1_feats)
        conv4_2_feats = Convolution2D(512, (3, 3), activation='relu')(zeropad4_2_feats)
        zeropad4_3_feats = ZeroPadding2D((1, 1))(conv4_2_feats)
        conv4_3_feats = Convolution2D(512, (3, 3), activation='relu')(zeropad4_3_feats)
        pool4_feats = (MaxPooling2D((2, 2), strides=(2, 2)))(conv4_3_feats)

        zeropad5_1_feats = ZeroPadding2D((1, 1), input_shape=(224, 224, 3))(pool4_feats)
        conv5_1_feats = Convolution2D(512, (3, 3), activation='relu')(zeropad5_1_feats)
        zeropad5_2_feats = ZeroPadding2D((1, 1))(conv5_1_feats)
        conv5_2_feats = Convolution2D(512, (3, 3), activation='relu')(zeropad5_2_feats)
        zeropad5_3_feats = ZeroPadding2D((1, 1))(conv5_2_feats)
        conv5_3_feats = Convolution2D(512, (3, 3), activation='relu')(zeropad5_3_feats)
        pool5_feats = (MaxPooling2D((2, 2), strides=(2, 2)))(conv5_3_feats)

        fully1_feats = (Convolution2D(4096, (7, 7), activation='relu'))(pool5_feats)
        dropout1_feats = Dropout(0.5)(fully1_feats)
        fully2_feats = (Convolution2D(4096, (1, 1), activation='relu'))(dropout1_feats)
        dropout1_feats = Dropout(0.5)(fully2_feats)
        fully3_feats = (Convolution2D(2622, (1, 1), activation='relu'))(dropout1_feats)
        dropout1_feats = Flatten()(fully3_feats)
        softmax_feats = Activation('softmax')(dropout1_feats)

        model = Model(input_feats, softmax_feats)
        model.load_weights('vgg_face_weights.h5')

        for layer in model.layers:
            layer.trainable = False

        self.cnn_layer = pool4_feats  #1
        # self.cnn_layer = Reshape((196, 512))(pool4_feats)  #2

        # 512 앞단: pool4_feats, 512 뒷단: conv5_3_feats
        self.conv_feats = Reshape((196, 512))(conv5_3_feats)
        self.num_ctx = 196
        self.dim_ctx = 512
        # self.emotion_output = dense_feats3

    def build_rnn(self):
        """ Build the RNN. """
        print("Building the RNN...")
        config = self.config

        # Setup the placeholders
        if self.is_train:
            contexts = self.conv_feats
            if config.data == 'DISFA':
                emotions = tf.placeholder(
                    dtype=tf.int64,
                    shape=[config.batch_size, 6])
            else:
                emotions = tf.placeholder(
                    dtype=tf.int64,
                    shape=[config.batch_size, 7])

            sentences = tf.placeholder(
                dtype = tf.int32,
                shape = [config.batch_size, config.max_caption_length])
            masks = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, config.max_caption_length])
        else:
            if config.data == 'DISFA':
                emotions = tf.placeholder(
                    dtype=tf.int64,
                    shape=[config.batch_size, 6])
            else:
                emotions = tf.placeholder(
                    dtype=tf.int64,
                    shape=[config.batch_size, 7])
            contexts = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, self.num_ctx, self.dim_ctx])
            last_memory = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, config.num_lstm_units])
            last_output = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, config.num_lstm_units])
            last_word = tf.placeholder(
                dtype = tf.int32,
                shape = [config.batch_size])

        # Setup the word embedding
        with tf.variable_scope("word_embedding"):
            embedding_matrix = tf.get_variable(
                name = 'weights',
                shape = [config.vocabulary_size, config.dim_embedding],
                initializer = self.nn.fc_kernel_initializer,
                regularizer = self.nn.fc_kernel_regularizer,
                trainable = self.is_train)

        # Setup the LSTM
        lstm = tf.nn.rnn_cell.LSTMCell(
            config.num_lstm_units,
            initializer = self.nn.fc_kernel_initializer)
        if self.is_train:
            lstm = tf.nn.rnn_cell.DropoutWrapper(
                lstm,
                input_keep_prob = 1.0-config.lstm_drop_rate,
                output_keep_prob = 1.0-config.lstm_drop_rate,
                state_keep_prob = 1.0-config.lstm_drop_rate)

        # Initialize the LSTM using the mean context
        with tf.variable_scope("initialize"):
            context_mean = tf.reduce_mean(self.conv_feats, axis = 1)
            initial_memory, initial_output = self.initialize(context_mean)
            initial_state = initial_memory, initial_output

        # Prepare to run
        predictions = []
        if self.is_train:
            alphas = []
            cross_entropies = []
            predictions_correct = []
            num_steps = config.max_caption_length
            last_output = initial_output
            last_memory = initial_memory
            last_word = tf.zeros([config.batch_size], tf.int32)
        else:
            num_steps = config.max_caption_length ###????? 1????
        last_state = last_memory, last_output
        total_output = []
        # Generate the words one by one
        for idx in range(num_steps):
            # Attention mechanism
            with tf.variable_scope("attend"):
                alpha = self.attend(contexts, last_output)
                context = tf.reduce_sum(contexts*tf.expand_dims(alpha, 2),
                                        axis = 1)
                if self.is_train:
                    tiled_masks = tf.tile(tf.expand_dims(masks[:, idx], 1),
                                         [1, self.num_ctx])
                    masked_alpha = alpha * tiled_masks
                    alphas.append(tf.reshape(masked_alpha, [-1]))

            # Embed the last word
            with tf.variable_scope("word_embedding"):
                word_embed = tf.nn.embedding_lookup(embedding_matrix,
                                                    last_word)
            # Apply the LSTM
            with tf.variable_scope("lstm"):
                current_input = tf.concat([context, word_embed], 1)
                output, state = lstm(current_input, last_state)
                memory, _ = state

            if config.is_train == True:
                total_output.append(output)

            # Decode the expanded output of LSTM into a word
            with tf.variable_scope("decode"):
                expanded_output = tf.concat([output,
                                             context,
                                             word_embed],
                                             axis = 1)

                # print(tf.shape(expanded_output))
                logits = self.decode(expanded_output)
                probs = tf.nn.softmax(logits)
                prediction = tf.argmax(logits, 1)
                predictions.append(prediction)

            # Compute the loss for this step, if necessary
            if self.is_train:
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels = sentences[:, idx],
                    logits = logits)
                masked_cross_entropy = cross_entropy * masks[:, idx]
                cross_entropies.append(masked_cross_entropy)

                ground_truth = tf.cast(sentences[:, idx], tf.int64)
                prediction_correct = tf.where(
                    tf.equal(prediction, ground_truth),
                    tf.cast(masks[:, idx], tf.float32),
                    tf.cast(tf.zeros_like(prediction), tf.float32))
                predictions_correct.append(prediction_correct)

                last_output = output
                last_memory = memory
                last_state = state
                last_word = sentences[:, idx]

            tf.get_variable_scope().reuse_variables()

        if config.is_train == False:
            for _ in range(config.max_caption_length):
                total_output.append(output)

        total_output = tf.stack(total_output, axis=1)
        print('lstm output shape: ', last_output.get_shape())
        self.build_emotion(total_output)  # 첫번째 시도: last_output,  두번째 시도: total_output

        # Compute the final loss, if necessary
        if self.is_train:
            # emotion
            if self.config.cnn == 'vgg16_emotion' or self.config.cnn == 'vggface_emotion':
                emotion_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=emotions[:],
                                                                                   logits=self.emotion_output)
                # print(np.shape(self.emotion_output))
                emotion_cross_entropy_loss = tf.reduce_mean(emotion_cross_entropy)

            cross_entropies = tf.stack(cross_entropies, axis = 1)
            cross_entropy_loss = tf.reduce_sum(cross_entropies) \
                                 / tf.reduce_sum(masks)

            alphas = tf.stack(alphas, axis = 1)
            alphas = tf.reshape(alphas, [config.batch_size, self.num_ctx, -1])
            attentions = tf.reduce_sum(alphas, axis = 2)
            # print(np.shape(attentions))

            diffs = tf.ones_like(attentions) - attentions
            attention_loss = config.attention_loss_factor \
                             * tf.nn.l2_loss(diffs) \
                             / (config.batch_size * self.num_ctx)

            reg_loss = tf.losses.get_regularization_loss()

            if self.config.cnn == 'vgg16_emotion' or self.config.cnn == 'vggface_emotion':
                total_loss = cross_entropy_loss + attention_loss + reg_loss + 0.00001 * emotion_cross_entropy_loss
            else:
                total_loss = cross_entropy_loss + attention_loss + reg_loss

            predictions_correct = tf.stack(predictions_correct, axis = 1)
            accuracy = tf.reduce_sum(predictions_correct) \
                       / tf.reduce_sum(masks)

        self.contexts = contexts
        if self.is_train:
            self.emotions = emotions
            self.sentences = sentences
            self.masks = masks
            self.total_loss = total_loss
            if self.config.cnn == 'vgg16_emotion' or self.config.cnn == 'vggface_emotion':
                self.emotion_cross_entropy_loss = emotion_cross_entropy_loss
            self.cross_entropy_loss = cross_entropy_loss
            self.attention_loss = attention_loss
            self.reg_loss = reg_loss
            self.accuracy = accuracy
            self.attentions = attentions

            self.initial_memory = initial_memory
            self.initial_output = initial_output
            self.last_memory = last_memory
            self.last_output = last_output
            self.last_word = last_word
            self.memory = memory
            self.output = output
            self.probs = probs

        else:
            self.initial_memory = initial_memory
            self.initial_output = initial_output
            self.last_memory = last_memory
            self.last_output = last_output
            self.last_word = last_word
            self.memory = memory
            self.output = output
            self.probs = probs

            self.emotions = emotions

        print("RNN built.")

    def build_emotion(self, rnn_layer):
        config = self.config

        print('Building Emotion...')
        print('cnn layer shape', self.cnn_layer.get_shape())
        with tf.variable_scope("emotion", reuse = tf.AUTO_REUSE):
            flatten_rnn_feats = Flatten()(rnn_layer)
            rnn_feats = Dense(49 * 64, activation='relu')(flatten_rnn_feats)
            print('rnn_feats_shape', rnn_feats.get_shape())

            rnn_feats = Reshape((7, 7, 64))(rnn_feats)
            rnn_feats=Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(rnn_feats))
            print('rnn_feats_upsample', rnn_feats.get_shape())

            rnn_feats = Convolution2D(256, 3, padding = 'same', kernel_initializer = 'he_normal', activation='relu')(rnn_feats)
            cnn_layer = Convolution2D(256, 3, padding = 'same', kernel_initializer = 'he_normal', activation='relu')(self.cnn_layer)

            print('rnn_feats_shape', rnn_feats.get_shape())

            rnn_feats_temp = tf.placeholder_with_default(rnn_feats, (None, 14, 14, 256), name='temp')
            concat_feats = concatenate([cnn_layer, rnn_feats_temp], axis=3)
            print('concat_feats', concat_feats.get_shape())
            # flatten_feats = Flatten()(concat_feats)
            # print('flatten_feats', flatten_feats.get_shape())
            dense_feats1 = Convolution2D(512, 3, padding = 'same', kernel_initializer = 'he_normal', activation='relu')(concat_feats)
            dense_feats1= BatchNormalization()(dense_feats1)
            # assert concat_feats.shape.as_list() == [None, 1024]
            print('dense_feats1', dense_feats1.get_shape())

            # dropout2_1_feats = Dropout(0.2)(dense_feats1)
            dense_feats2 = Convolution2D(256, 3, padding = 'same', kernel_initializer = 'he_normal', activation='relu')(dense_feats1)
            dense_feats2= BatchNormalization()(dense_feats2)

            print('dense_feats2', dense_feats2.get_shape())

            flatten_feats = Flatten()(dense_feats2)
            dense_feats3 = Dense(128, activation='relu')(flatten_feats)

            dropout_feats = Dropout(0.2)(dense_feats3)

            if config.data == 'DISFA':
                dense_feats3 = Dense(6, activation='softmax')(dropout_feats)
            else:
                dense_feats3 = Dense(7, activation='softmax')(dropout_feats)

        self.emotion_output = dense_feats3


    def initialize(self, context_mean):
        """ Initialize the LSTM using the mean context. """
        config = self.config
        context_mean = self.nn.dropout(context_mean)

        if config.num_initalize_layers == 1:
            # use 1 fc layer to initialize
            memory = self.nn.dense(context_mean,
                                   units = config.num_lstm_units,
                                   activation = None,
                                   name = 'fc_a')
            output = self.nn.dense(context_mean,
                                   units = config.num_lstm_units,
                                   activation = None,
                                   name = 'fc_b')
        else:
            # use 2 fc layers to initialize
            temp1 = self.nn.dense(context_mean,
                                  units = config.dim_initalize_layer,
                                  activation = tf.tanh,
                                  name = 'fc_a1')
            temp1 = self.nn.dropout(temp1)
            memory = self.nn.dense(temp1,
                                   units = config.num_lstm_units,
                                   activation = None,
                                   name = 'fc_a2')

            temp2 = self.nn.dense(context_mean,
                                  units = config.dim_initalize_layer,
                                  activation = tf.tanh,
                                  name = 'fc_b1')
            temp2 = self.nn.dropout(temp2)
            output = self.nn.dense(temp2,
                                   units = config.num_lstm_units,
                                   activation = None,
                                   name = 'fc_b2')
        return memory, output

    def attend(self, contexts, output):
        """ Attention Mechanism. """
        config = self.config
        reshaped_contexts = tf.reshape(contexts, [-1, self.dim_ctx])
        reshaped_contexts = self.nn.dropout(reshaped_contexts)
        output = self.nn.dropout(output)
        if config.num_attend_layers == 1:
            # use 1 fc layer to attend
            logits1 = self.nn.dense(reshaped_contexts,
                                    units = 1,
                                    activation = None,
                                    use_bias = False,
                                    name = 'fc_a')
            logits1 = tf.reshape(logits1, [-1, self.num_ctx])
            logits2 = self.nn.dense(output,
                                    units = self.num_ctx,
                                    activation = None,
                                    use_bias = False,
                                    name = 'fc_b')
            logits = logits1 + logits2
        else:
            # use 2 fc layers to attend
            temp1 = self.nn.dense(reshaped_contexts,
                                  units = config.dim_attend_layer,
                                  activation = tf.tanh,
                                  name = 'fc_1a')
            temp2 = self.nn.dense(output,
                                  units = config.dim_attend_layer,
                                  activation = tf.tanh,
                                  name = 'fc_1b')
            temp2 = tf.tile(tf.expand_dims(temp2, 1), [1, self.num_ctx, 1])
            temp2 = tf.reshape(temp2, [-1, config.dim_attend_layer])
            temp = temp1 + temp2
            temp = self.nn.dropout(temp)
            logits = self.nn.dense(temp,
                                   units = 1,
                                   activation = None,
                                   use_bias = False,
                                   name = 'fc_2')
            logits = tf.reshape(logits, [-1, self.num_ctx])
        alpha = tf.nn.softmax(logits)
        return alpha

    def decode(self, expanded_output):
        """ Decode the expanded output of the LSTM into a word. """
        config = self.config
        expanded_output = self.nn.dropout(expanded_output)
        if config.num_decode_layers == 1:
            # use 1 fc layer to decode
            logits = self.nn.dense(expanded_output,
                                   units = config.vocabulary_size,
                                   activation = None,
                                   name = 'fc')
        else:
            # use 2 fc layers to decode
            temp = self.nn.dense(expanded_output,
                                 units = config.dim_decode_layer,
                                 activation = tf.tanh,
                                 name = 'fc_1')
            temp = self.nn.dropout(temp)
            logits = self.nn.dense(temp,
                                   units = config.vocabulary_size,
                                   activation = None,
                                   name = 'fc_2')
        return logits

    def build_optimizer(self):
        """ Setup the optimizer and training operation. """
        config = self.config

        learning_rate = tf.constant(config.initial_learning_rate)
        if config.learning_rate_decay_factor < 1.0:
            def _learning_rate_decay_fn(learning_rate, global_step):
                return tf.train.exponential_decay(
                    learning_rate,
                    global_step,
                    decay_steps = config.num_steps_per_decay,
                    decay_rate = config.learning_rate_decay_factor,
                    staircase = True)
            learning_rate_decay_fn = _learning_rate_decay_fn
        else:
            learning_rate_decay_fn = None

        with tf.variable_scope('optimizer', reuse = tf.AUTO_REUSE):
            if config.optimizer == 'Adam':
                optimizer = tf.train.AdamOptimizer(
                    learning_rate = config.initial_learning_rate,
                    beta1 = config.beta1,
                    beta2 = config.beta2,
                    epsilon = config.epsilon
                    )
            elif config.optimizer == 'RMSProp':
                optimizer = tf.train.RMSPropOptimizer(
                    learning_rate = config.initial_learning_rate,
                    decay = config.decay,
                    momentum = config.momentum,
                    centered = config.centered,
                    epsilon = config.epsilon
                )
            elif config.optimizer == 'Momentum':
                optimizer = tf.train.MomentumOptimizer(
                    learning_rate = config.initial_learning_rate,
                    momentum = config.momentum,
                    use_nesterov = config.use_nesterov
                )
            else:
                optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate = config.initial_learning_rate
                )

            opt_op = tf.contrib.layers.optimize_loss(
                loss = self.total_loss,
                global_step = self.global_step,
                learning_rate = learning_rate,
                optimizer = optimizer,
                # clip_gradients = config.clip_gradients,
                learning_rate_decay_fn = learning_rate_decay_fn)

        self.opt_op = opt_op

    def build_summary(self):
        """ Build the summary (for TensorBoard visualization). """
        with tf.name_scope("variables"):
            for var in tf.trainable_variables():
                with tf.name_scope(var.name[:var.name.find(":")]):
                    self.variable_summary(var)

        with tf.name_scope("metrics"):
            tf.summary.scalar("cross_entropy_loss", self.cross_entropy_loss)
            if self.config.cnn == 'vgg16_emotion' or self.config.cnn == 'vggface_emotion':
                tf.summary.scalar("emotion_cross_entropy_loss", self.emotion_cross_entropy_loss)
            tf.summary.scalar("attention_loss", self.attention_loss)
            tf.summary.scalar("reg_loss", self.reg_loss)
            tf.summary.scalar("total_loss", self.total_loss)
            tf.summary.scalar("accuracy", self.accuracy)

        with tf.name_scope("attentions"):
            self.variable_summary(self.attentions)

        self.summary = tf.summary.merge_all()

    def variable_summary(self, var):
        """ Build the summary for a variable. """
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


##### augmentation

def _augment(images,
            resize=None,  # (width, height) tuple or None
            horizontal_flip=False,
            vertical_flip=False,
            rotate=5.0,  # Maximum rotation angle in degrees
            crop_probability=0.4,  # How often we do crops
            crop_min_percent=0.90,  # Minimum linear dimension of a crop
            crop_max_percent=1.,  # Maximum linear dimension of a crop
            mixup=0):  # Mixup coeffecient, see https://arxiv.org/abs/1710.09412.pdf

    if resize is not None:
        images = tf.image.resize_bilinear(images, resize)
    # My experiments showed that casting on GPU improves training performance
    if images.dtype != tf.float32:
        images = tf.image.convert_image_dtype(images, dtype=tf.float32)
        images = tf.subtract(images, 0.5)
        images = tf.multiply(images, 2.0)

    with tf.name_scope('augmentation'):
        shp = tf.shape(images)
        shp2 = images.get_shape().as_list()
        batch_size, height, width = shp[0], shp2[1], shp2[2]

        # The list of affine transformations that our image will go under.
        # Every element is Nx8 tensor, where N is a batch size.
        transforms = []
        identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        if horizontal_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            flip_transform = tf.convert_to_tensor(
                [-1., 0., width, 0., 1., 0., 0., 0.], dtype=tf.float32)
            transforms.append(
                tf.where(coin,
                         tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if vertical_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            flip_transform = tf.convert_to_tensor(
                [1, 0, 0, 0, -1, height, 0, 0], dtype=tf.float32)
            transforms.append(
                tf.where(coin,
                         tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if rotate > 0:
            angle_rad = rotate / 180 * math.pi
            angles = tf.random_uniform([batch_size], -angle_rad, angle_rad)

            transforms.append(
                tf.contrib.image.angles_to_projective_transforms(
                    angles, height, width))

        if crop_probability > 0:
            crop_pct = tf.random_uniform([batch_size], crop_min_percent,
                                         crop_max_percent)
            left = tf.random_uniform([batch_size], 0, width * (1 - crop_pct))
            top = tf.random_uniform([batch_size], 0, height * (1 - crop_pct))

            crop_transform = tf.stack([
                crop_pct,
                tf.zeros([batch_size]), top,
                tf.zeros([batch_size]), crop_pct, left,
                tf.zeros([batch_size]),
                tf.zeros([batch_size])
            ], 1)

            coin = tf.less(
                tf.random_uniform([batch_size], 0, 1.0), crop_probability)
            transforms.append(
                tf.where(coin, crop_transform,
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if transforms:
            images = tf.contrib.image.transform(
                images,
                tf.contrib.image.compose_transforms(*transforms),
                interpolation='BILINEAR')  # or 'NEAREST'

            resized_image = tf.image.central_crop(images, float(205) / 224)
            images = tf.image.resize_images(resized_image, [224, 224],
                                           method=tf.image.ResizeMethod.BILINEAR, align_corners=False)

        def cshift(values):  # Circular shift in batch dimension
            return tf.concat([values[-1:, ...], values[:-1, ...]], 0)

        if mixup > 0:
            mixup = 1.0 * mixup  # Convert to float, as tf.distributions.Beta requires floats.
            beta = tf.distributions.Beta(mixup, mixup)
            lam = beta.sample(batch_size)
            ll = tf.expand_dims(tf.expand_dims(tf.expand_dims(lam, -1), -1), -1)
            images = ll * images + (1 - ll) * cshift(images)

    return images

def _largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.
    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow
    Converted to Python by Aaron Snoswell
    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """
    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    ww = bb_w - 2 * x
    hh = bb_h - 2 * y

    return (w-ww)/2, (h-hh)/2