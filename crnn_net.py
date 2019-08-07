import os
import time
import numpy as np
import tensorflow as tf
import config
from scipy.misc import imread, imresize, imsave
from tensorflow.contrib import rnn

from data_loader import DataLoader
from utils import sparse_tuple_from, resize_image, label_to_array, ground_truth_to_word, levenshtein

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class CRNN(object):
    def __init__(self, batch_size, model_path, examples_path, max_image_width, train_test_ratio, restore):
        self.step = 0
        self.__model_path = model_path
        self.__save_path = os.path.join(model_path, 'ckp')

        self.__restore = restore

        self.__training_name = str(int(time.time()))
        self.__session = tf.Session()

        # Building graph
        with self.__session.as_default():
            (
                self.__inputs,
                self.__targets,
                self.__seq_len,
                self.__logits,
                self.__decoded,
                self.__predict,
                self.__optimizer,
                self.__acc,
                self.__cost,
                self.__max_char_count,
                self.__inits
            ) = self.crnn()
            for __init in self.__inits:
                __init.run()

        with self.__session.as_default():
            self.__saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
            if self.__restore:
                print('Restoring')
                ckpt = tf.train.latest_checkpoint(self.__model_path)
                if ckpt:
                    print('Checkpoint is valid')
                    self.step = int(ckpt.split('-')[1])
                    self.__saver.restore(self.__session, ckpt)

        # Creating data_manager
        self.__data_loader = DataLoader(examples_path, batch_size, max_image_width, self.__max_char_count)

    def crnn(self):
        def BidirectionnalRNN(inputs, seq_len):
            """
                Bidirectionnal LSTM Recurrent Neural Network part
            """

            with tf.variable_scope(None, default_name="bidirectional-rnn-1"):
                # Forward
                lstm_fw_cell_1 = rnn.BasicLSTMCell(256)
                # Backward
                lstm_bw_cell_1 = rnn.BasicLSTMCell(256)

                inter_output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_1, lstm_bw_cell_1, inputs, seq_len, dtype=tf.float32)

                inter_output = tf.concat(inter_output, 2)

            with tf.variable_scope(None, default_name="bidirectional-rnn-2"):
                # Forward
                lstm_fw_cell_2 = rnn.BasicLSTMCell(256)
                # Backward
                lstm_bw_cell_2 = rnn.BasicLSTMCell(256)

                outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_2, lstm_bw_cell_2, inter_output, seq_len, dtype=tf.float32)

                outputs = tf.concat(outputs, 2)


            return outputs

        def CNN(inputs):
            """
                Convolutionnal Neural Network part
            """

            # 64 / 3 x 3 / 1 / 1
            conv1 = tf.layers.conv2d(inputs=inputs, filters = 64, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)

            # 2 x 2 / 1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            # 128 / 3 x 3 / 1 / 1
            conv2 = tf.layers.conv2d(inputs=pool1, filters = 128, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)

            # 2 x 2 / 1
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

            # 256 / 3 x 3 / 1 / 1
            conv3 = tf.layers.conv2d(inputs=pool2, filters = 256, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)

            # Batch normalization layer
            bnorm1 = tf.layers.batch_normalization(conv3)

            # 256 / 3 x 3 / 1 / 1
            conv4 = tf.layers.conv2d(inputs=bnorm1, filters = 256, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)

            # 1 x 2 / 1
            pool3 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=[1, 2], padding="same")

            # 512 / 3 x 3 / 1 / 1
            conv5 = tf.layers.conv2d(inputs=pool3, filters = 512, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)

            # Batch normalization layer
            bnorm2 = tf.layers.batch_normalization(conv5)

            # 512 / 3 x 3 / 1 / 1
            conv6 = tf.layers.conv2d(inputs=bnorm2, filters = 512, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)

            # 1 x 2 / 2
            pool4 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=[1, 2], padding="same")

            # 512 / 2 x 2 / 1 / 0
            conv7 = tf.layers.conv2d(inputs=pool4, filters = 512, kernel_size = (2, 2), padding = "valid", activation=tf.nn.relu)

            return conv7

        label_text = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(tf.constant(config.ALPHABET_INDEX, dtype=tf.int64),
                                                        tf.constant(config.ALPHABET, dtype=tf.string)),
            default_value=''
        )
        text_label = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(tf.constant(config.ALPHABET, dtype=tf.string),
                                                        tf.constant(config.ALPHABET_INDEX, dtype=tf.int32)),
            default_value=-1
        )

        inputs = tf.placeholder(tf.float32, [None, 100, 32, 1], name='inputs')
        targets = tf.sparse_placeholder(tf.int32, name='targets')
        batch_size = tf.shape(inputs)[0]

        cnn_output = CNN(inputs)
        cnn_output_shape = cnn_output.get_shape().as_list()

        reshaped_cnn_output = tf.reshape(cnn_output, [-1, cnn_output_shape[1] * cnn_output_shape[2], 512])

        max_char_count = reshaped_cnn_output.get_shape().as_list()[1]

        sequence_length = tf.fill([batch_size], value=max_char_count, name='seq_len')

        crnn_model = BidirectionnalRNN(reshaped_cnn_output, sequence_length)

        logits = tf.reshape(crnn_model, [-1, 512])

        W = tf.Variable(tf.truncated_normal([512, config.NUM_CLASSES], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0., shape=[config.NUM_CLASSES]), name="b")

        logits = tf.matmul(logits, W) + b

        logits = tf.reshape(logits, [-1, max_char_count, config.NUM_CLASSES])

        # Final layer, the output of the BLSTM
        logits = tf.transpose(logits, (1, 0, 2))

        # Loss and cost calculation
        loss = tf.nn.ctc_loss(targets, logits, sequence_length)

        cost = tf.reduce_mean(loss)

        # Training step
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

        # The decoded answer
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, sequence_length, merge_repeated=False)

        dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value=-1)

        # The error rate
        acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

        table = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(tf.constant(config.ALPHABET_INDEX, dtype=tf.int64),
                                                        tf.constant(config.ALPHABET, dtype=tf.string)),
            default_value=''
        )

        predict_out = table.lookup(dense_decoded)

        inits = [tf.global_variables_initializer(), tf.tables_initializer()]

        return inputs, targets, sequence_length, logits, dense_decoded, predict_out, optimizer, acc, cost, max_char_count, inits

    def train(self, epoch_count):
        with self.__session.as_default():
            print('Training')
            for i in range(self.step, epoch_count + self.step):
                iter_loss = 0
                for batch_y, batch_dt, batch_x in self.__data_loader:
                    op, decoded, loss_value = self.__session.run(
                        [self.__optimizer, self.__decoded, self.__cost],
                        feed_dict={
                            self.__inputs: batch_x,
                            self.__targets: batch_dt
                        }
                    )

                    if i % 10 == 0:
                        for j in range(2):
                            print('true:[{}]  predict:[{}]'.format(batch_y[j], ground_truth_to_word(decoded[j])))

                    iter_loss += loss_value

                self.__saver.save(
                    self.__session,
                    self.__save_path,
                    global_step=self.step
                )

                print('[{}] epoch loss: {}'.format(self.step, iter_loss))

                self.step += 1
        return None

    def test(self):
        with self.__session.as_default():
            print('Testing')
            success_count = 0
            total_count = 0
            for batch_y, _, batch_x in self.__data_loader:
                decoded = self.__session.run(
                    self.__decoded,
                    feed_dict={
                        self.__inputs: batch_x,
                        self.__seq_len: [self.__max_char_count] * self.__data_loader.batch_size
                    }
                )
                for i, y in enumerate(batch_y):
                    total_count += 1
                    true_label = batch_y[i]
                    predict = ground_truth_to_word(decoded[i])
                    if true_label == predict:
                        print(true_label + '\t' + predict + '\t' + 'predict successfully')
                        success_count += 1
                    else:
                        print(true_label + '\t' + predict + '\t' + 'predict failed')
            print('准确率: ' + str(success_count/total_count))

        return None