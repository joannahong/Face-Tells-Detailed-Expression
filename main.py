#!/usr/bin/python
import tensorflow as tf
from keras import backend as K

from config import Config
from model import CaptionGenerator
from dataset import prepare_train_data, prepare_eval_data#, prepare_test_data
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('phase', 'train',
                       'The phase can be train, eval or test')

tf.flags.DEFINE_boolean('load', False,
                        'Turn on to load a pretrained model from either \
                        the latest checkpoint or a specified file')

tf.flags.DEFINE_string('model_file', None,
                       'If sepcified, load a pretrained model from this file')

tf.flags.DEFINE_boolean('load_cnn', False,
                        'Turn on to load a pretrained CNN model')

tf.flags.DEFINE_string('cnn_model_file', './vgg16_no_fc.npy',
                       'The file containing a pretrained CNN model')

tf.flags.DEFINE_boolean('train_cnn', False,
                        'Turn on to train both CNN and RNN. \
                         Otherwise, only RNN is trained')

tf.flags.DEFINE_integer('beam_size', 3,
                        'The size of beam search for caption generation')

def main(argv):
    config = Config()
    config.phase = FLAGS.phase
    config.train_cnn = FLAGS.train_cnn
    config.beam_size = FLAGS.beam_size

    sess = tf.Session()
    K.set_session(sess)

    if FLAGS.phase == 'train':
        # training phase
        train_data = prepare_train_data(config)
        eval_data, vocabulary = prepare_eval_data(config)
        model = CaptionGenerator(config)
        sess.run(tf.global_variables_initializer())
        if FLAGS.load:
            model.load(sess, FLAGS.model_file)
        if FLAGS.load_cnn:
            model.load_cnn(sess, FLAGS.cnn_model_file)
        tf.get_default_graph().finalize()
        tf.set_random_seed(seed=0)
        model.train(sess, train_data, eval_data, vocabulary)

    elif FLAGS.phase == 'eval':
        # evaluation phase
        data, vocabulary = prepare_eval_data(config)
        model = CaptionGenerator(config)
        sess.run(tf.global_variables_initializer())
        model.load(sess, FLAGS.model_file)
        tf.get_default_graph().finalize()
        model.eval(sess, data, vocabulary)

    # elif FLAGS.phase == 'both':
    #     # training with evaluation
    #     train_data = prepare_train_data(config)
    #     eval_data, vocabulary = prepare_eval_data(config)
    #     model = CaptionGenerator(config)
    #     sess.run(tf.global_variables_initializer())
    #     if FLAGS.load:
    #         model.load(sess, FLAGS.model_file)
    #     if FLAGS.load_cnn:
    #         model.load_cnn(sess, FLAGS.cnn_model_file)
    #     tf.get_default_graph().finalize()
    #     model.trainweval(sess, train_data, eval_data, vocabulary)
    #
    # else:
    #     # testing phase
    #     data, vocabulary = prepare_test_data(config)
    #     model = CaptionGenerator(config)
    #     model.load(sess, FLAGS.model_file)
    #     tf.get_default_graph().finalize()
    #     model.test(sess, data, vocabulary)

    sess.close()

if __name__ == '__main__':
    tf.app.run()
