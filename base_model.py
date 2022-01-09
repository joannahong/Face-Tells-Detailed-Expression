import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import matplotlib
from textwrap import wrap
import cPickle as pickle
import copy
import json
from tqdm import tqdm
from keras import backend as K

from utils.nn import NN
from utils.coco.coco import COCO
# from utils.coco.pycocoevalcap.eval import COCOEvalCap
from utils.coco.pycocoevalcap.ckeval import EvalCap
from utils.coco.pycocoevalcap.classify import Classify
from misc import EmotionLoader, ImageLoader, CaptionData, TopN
matplotlib.use('agg')
plt.switch_backend('agg')
class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.is_train = True if config.phase == 'train' or config.phase == 'both' else False
        self.train_cnn = self.is_train and config.train_cnn
        self.emotion_loader = EmotionLoader()
        self.image_loader = ImageLoader()
        self.image_shape = [224, 224, 3]
        self.nn = NN(config)
        self.global_step = tf.Variable(0,
                                       name = 'global_step',
                                       trainable = False)
        if self.is_train:
            print('Training begins...')
        self.build()

    def build(self):
        raise NotImplementedError()

    def train_real(self, sess, train_data):
        print("Training the model...")
        config = self.config
        if not os.path.exists(config.summary_dir):
            os.mkdir(config.summary_dir)
        train_writer = tf.summary.FileWriter(config.summary_dir,
                                             sess.graph)

        for _ in tqdm(list(range(config.num_epochs)), desc='epoch'):
            for _ in tqdm(list(range(train_data.num_batches)), desc='batch'):
                batch = train_data.next_batch()
                emotion_files, image_files, sentences, masks = batch
                emotions = self.emotion_loader.load_emotions(emotion_files)
                images = self.image_loader.load_images(image_files)
                # print(emotions)
                if self.config.cnn == 'vgg16_emotion' or self.config.cnn == 'vggface_emotion':
                    feed_dict = {self.emotions: emotions,
                                 self.images: images,
                                 self.sentences: sentences,
                                 self.masks: masks}
                    _, summary, global_step = sess.run([self.opt_op,
                                                        self.summary,
                                                        self.global_step],
                                                       feed_dict=feed_dict)
                else:
                    feed_dict = {self.images: images,
                                 self.sentences: sentences,
                                 self.masks: masks}
                    _, summary, global_step = sess.run([self.opt_op,
                                                        self.summary,
                                                        self.global_step],
                                                        feed_dict=feed_dict)

                if (global_step+1) % config.save_period == 0:
                    #self.eval(sess, eval_data, vocabulary, config.save_period)
                    self.save(sess)
                train_writer.add_summary(summary, global_step)
            train_data.reset()

        self.save(sess)
        train_writer.close()
        print("Training complete.")

    def train(self, sess, train_data, eval_data, vocabulary):
        print("Training the model...")
        config = self.config
        if not os.path.exists(config.summary_dir):
            os.mkdir(config.summary_dir)
        train_writer = tf.summary.FileWriter(config.summary_dir,
                                             sess.graph)
        if not os.path.exists(config.save_dir):
            os.mkdir(config.save_dir)
        best_emo_acc = 0
        for _ in tqdm(list(range(config.num_epochs)), desc='epoch'):
            for _ in tqdm(list(range(train_data.num_batches)), desc='batch'):
                batch = train_data.next_batch()
                emotion_files, image_files, sentences, masks = batch
                if config.data == 'CK+':
                    emotions = self.emotion_loader.load_emotions(emotion_files)
                else:
                    emotions = self.emotion_loader.load_data(emotion_files)
                images = self.image_loader.load_images(image_files)
                # print(emotions)
                if self.config.cnn == 'vgg16_emotion' or self.config.cnn == 'vggface_emotion':
                    feed_dict = {self.emotions: emotions,
                                 self.images: images,
                                 self.sentences: sentences,
                                 self.masks: masks}
                    _, summary, global_step = sess.run([self.opt_op,
                                                        self.summary,
                                                        self.global_step],
                                                       feed_dict=feed_dict)
                else:
                    feed_dict = {self.images: images,
                                 self.sentences: sentences,
                                 self.masks: masks}
                    _, summary, global_step = sess.run([self.opt_op,
                                                        self.summary,
                                                        self.global_step],
                                                       feed_dict=feed_dict)

                if self.config.cnn == 'vgg16_emotion' or self.config.cnn == 'vggface_emotion':
                    if (global_step + 1) % config.eval_period == 0:
                        # evaluation
                        results, results_idx, emo_results, emo_gts = [], [], [], []
                        idx = 0
                        for k in tqdm(list(range(eval_data.num_batches)), desc='batch'):
                            emotion_files, batch = eval_data.next_batch()
                            image_files = batch

                            emotion_data = self.beam_search(sess, emotion_files, image_files, vocabulary, sentences, masks)
                            fake_cnt = 0 if k < eval_data.num_batches - 1 \
                                else eval_data.fake_count

                            if config.data == 'CK+':
                                emo = self.emotion_loader.load_emotions(emotion_files)
                            else:
                                emo = self.emotion_loader.load_data(emotion_files)

                            emo_results.append(emotion_data)
                            emo_gts.append(emo)

                        classifier = Classify(config.emotion_file, config.action_unit_file, config.intensity_file)
                        emo_acc = classifier.emotionCheck(emo_results, emo_gts)
                        if emo_acc > best_emo_acc:
                            best_emo_acc = emo_acc
                        print('best emotion acc: {:.5f} %'.format(best_emo_acc))
                        eval_data.reset()
                print('************global step***************', global_step)
                if global_step > config.save_threshold:
                    if (global_step + 1) % config.save_period == 0:
                        # self.eval(sess, eval_data, vocabulary, config.save_period)
                        self.save(sess)
                        print('saving checkpoint')
                train_writer.add_summary(summary, global_step)
            train_data.reset()

        self.save(sess)
        train_writer.close()
        print("Training complete.")

    def eval(self, sess, eval_data, vocabulary):
        print("Evaluating the model ...")

        config = self.config

        results = []
        results_idx = []
        emo_results = []
        emo_gts = []
        if not os.path.exists(config.eval_result_dir):
            os.mkdir(config.eval_result_dir)

        # Generate the captions for the images
        idx = 0
        for k in tqdm(list(range(eval_data.num_batches)), desc='batch'):
            emotion_files, batch = eval_data.next_batch()
            image_files = batch

            if self.config.cnn == 'vgg16_emotion' or self.config.cnn == 'vggface_emotion':
                emotion_data, caption_data = self.beam_search(sess, emotion_files, image_files, vocabulary)
                fake_cnt = 0 if k<eval_data.num_batches-1 \
                             else eval_data.fake_count

                if config.data == 'CK+':
                    emo = self.emotion_loader.load_emotions(emotion_files)
                else:
                    emo = self.emotion_loader.load_data(emotion_files)
                emo_results.append(emotion_data)
                emo_gts.append(emo)

            caption_data = self.beam_search_vggonly(sess, emotion_files, image_files, vocabulary)
            fake_cnt = 0 if k < eval_data.num_batches - 1 \
                else eval_data.fake_count

            for l in range(eval_data.batch_size-fake_cnt):
                word_idxs = caption_data[l][0].sentence
                score = caption_data[l][0].score
                caption = vocabulary.get_sentence(word_idxs)
                results_idx.append({'caption_idx': word_idxs})
                results.append({'caption': caption})
                idx += 1

                # Save the result in an image file, if requested
                if config.save_eval_result_as_image:
                    image_file = batch[l]
                    image_name = image_file.split(os.sep)[-1]
                    image_name = os.path.splitext(image_name)[0]
                    if config.server == True:
                        server_dir = self.config.img_dir
                        image_file = server_dir + image_file[1:]
                        img = plt.imread(image_file)
                    else:
                        img = plt.imread(image_file)
                    plt.imshow(img, cmap='gray')
                    plt.axis('off')
                    plt.title("\n".join(wrap(caption)), fontsize=8)
                    plt.savefig(os.path.join(config.eval_result_dir,
                                             image_name+'_result.jpg'))

        results = pd.DataFrame.from_dict(results)
        print(results)
        result_data_dir = config.eval_result_dir + 'results.csv'
        if os.path.exists(result_data_dir):
            os.remove(result_data_dir)
        results.to_csv(result_data_dir)

        # Evaluate these captions
        scorer = EvalCap(result_data_dir, config.annotation_file_eval)
        scorer.evaluate()



        # classifier = Classify(config.emotion_file, config.action_unit_file, config.intensity_file)
        # classifier.emotionCheck(emo_results, emo_gts)
        # classifier.emogenCheck(result_data_dir, config.annotation_file_eval)
        # classifier.auintCheck(result_data_dir, config.annotation_file_eval)

        # fp = open(config.eval_result_file, 'wb')
        # json.dump(results, fp)
        # fp.close()

        # # Evaluate these captions
        # eval_result_coco = eval_gt_coco.loadRes(config.eval_result_file)
        # scorer = COCOEvalCap(eval_gt_coco, eval_result_coco)
        # scorer.evaluate()
        print("Evaluation complete.")


    # def test(self, sess, test_data, vocabulary):
    #     """ Test the model using any given images. """
    #     print("Testing the model ...")
    #     config = self.config
    #
    #     if not os.path.exists(config.test_result_dir):
    #         os.mkdir(config.test_result_dir)
    #
    #     captions = []
    #     scores = []
    #
    #     # Generate the captions for the images
    #     for k in tqdm(list(range(test_data.num_batches)), desc='path'):
    #         batch = test_data.next_batch()
    #         caption_data = self.beam_search(sess, batch, vocabulary)
    #
    #         fake_cnt = 0 if k<test_data.num_batches-1 \
    #                      else test_data.fake_count
    #         for l in range(test_data.batch_size-fake_cnt):
    #             word_idxs = caption_data[l][0].sentence
    #             score = caption_data[l][0].score
    #             caption = vocabulary.get_sentence(word_idxs)
    #             captions.append(caption)
    #             scores.append(score)
    #
    #             # Save the result in an image file
    #             image_file = batch[l]
    #             image_name = image_file.split(os.sep)[-1]
    #             image_name = os.path.splitext(image_name)[0]
    #             img = plt.imread(image_file)
    #             plt.imshow(img)
    #             plt.axis('off')
    #             plt.title(caption)
    #             plt.savefig(os.path.join(config.test_result_dir,
    #                                      image_name+'_result.jpg'))
    #
    #     # Save the captions to a file
    #     results = pd.DataFrame({'image_files':test_data.image_files,
    #                             'caption':captions,
    #                             'prob':scores})
    #     results.to_csv(config.test_result_file)
    #     print("Testing complete.")

    def beam_search_vggonly(self, sess, emotion_files, image_files, vocabulary, sentences = None, masks = None):
        """Use beam search to generate the captions for a batch of images."""
        # Feed in the images to get the contexts and the initial LSTM states
        config = self.config
        if config.data == 'CK+':
            emotions = self.emotion_loader.load_emotions(emotion_files)
        else:
            emotions = self.emotion_loader.load_data(emotion_files)

        images = self.image_loader.load_images(image_files)

        if self.is_train == True:
            contexts, initial_memory, initial_output, last_output, emotion= sess.run(
                [self.conv_feats, self.initial_memory, self.initial_output, self.last_output, self.emotion_output],
                feed_dict={self.emotions: emotions, self.images: images, self.sentences: sentences, self.masks: masks})

            emotion_result = emotion.argmax(axis=1)

            return emotion_result

        elif self.is_train == False:

            # lstm+emotion
            contexts, initial_memory, initial_output = sess.run(
                [self.conv_feats, self.initial_memory, self.initial_output],
                feed_dict = {self.emotions: emotions, self.images: images})

            partial_caption_data = []
            complete_caption_data = []
            for k in range(config.batch_size):
                initial_beam = CaptionData(sentence = [],
                                           memory = initial_memory[k],
                                           output = initial_output[k],
                                           score = 1.0)
                partial_caption_data.append(TopN(config.beam_size))
                partial_caption_data[-1].push(initial_beam)
                complete_caption_data.append(TopN(config.beam_size))

            # Run beam search
            for idx in range(config.max_caption_length):
                partial_caption_data_lists = []
                for k in range(config.batch_size):
                    data = partial_caption_data[k].extract()
                    partial_caption_data_lists.append(data)
                    partial_caption_data[k].reset()
                num_steps = 1 if idx == 0 else config.beam_size
                for b in range(num_steps):
                    if idx == 0:
                        last_word = np.zeros((config.batch_size), np.int32)
                    else:
                        last_word = np.array([pcl[b].sentence[-1]
                                            for pcl in partial_caption_data_lists],
                                            np.int32)

                    last_memory = np.array([pcl[b].memory
                                            for pcl in partial_caption_data_lists],
                                            np.float32)
                    last_output = np.array([pcl[b].output
                                            for pcl in partial_caption_data_lists],
                                            np.float32)

                    memory, output, scores= sess.run(
                        [self.memory, self.output, self.probs],
                        feed_dict = {self.contexts: contexts,
                                     self.last_word: last_word,
                                     self.last_memory: last_memory,
                                     self.last_output: last_output})

                    # Find the beam_size most probable next words
                    for k in range(config.batch_size):
                        caption_data = partial_caption_data_lists[k][b]
                        words_and_scores = list(enumerate(scores[k]))
                        words_and_scores.sort(key=lambda x: -x[1])
                        words_and_scores = words_and_scores[0:config.beam_size+1]
                        # Append each of these words to the current partial caption
                        for w, s in words_and_scores:
                            sentence = caption_data.sentence + [w]
                            score = caption_data.score * s
                            beam = CaptionData(sentence,
                                               memory[k],
                                               output[k],
                                               score)
                            if vocabulary.words[w] == '.':
                                complete_caption_data[k].push(beam)
                            else:
                                partial_caption_data[k].push(beam)
            # # lstm + emotion
            # emotion = sess.run(self.emotion_output,
            #     feed_dict={self.contexts: contexts,
            #                self.last_word: last_word,
            #                self.last_memory: last_memory,
            #                self.last_output: last_output,
            #                self.images: images})
            #
            # emotion_result = emotion.argmax(axis=1)

            results = []
            for k in range(config.batch_size):
                if complete_caption_data[k].size() == 0:
                    complete_caption_data[k] = partial_caption_data[k]
                results.append(complete_caption_data[k].extract(sort=True))

            return results #, emotion_result

    def beam_search(self, sess, emotion_files, image_files, vocabulary, sentences=None, masks=None):
        """Use beam search to generate the captions for a batch of images."""
        # Feed in the images to get the contexts and the initial LSTM states
        config = self.config
        if config.data == 'CK+':
            emotions = self.emotion_loader.load_emotions(emotion_files)
        else:
            emotions = self.emotion_loader.load_data(emotion_files)

        images = self.image_loader.load_images(image_files)

        if self.is_train == True:
            contexts, initial_memory, initial_output, last_output, emotion = sess.run(
                [self.conv_feats, self.initial_memory, self.initial_output, self.last_output, self.emotion_output],
                feed_dict={self.emotions: emotions, self.images: images, self.sentences: sentences, self.masks: masks})

            emotion_result = emotion.argmax(axis=1)

            return emotion_result

        elif self.is_train == False:

            # lstm+emotion
            contexts, initial_memory, initial_output = sess.run(
                [self.conv_feats, self.initial_memory, self.initial_output],
                feed_dict={self.emotions: emotions, self.images: images})

            partial_caption_data = []
            complete_caption_data = []
            for k in range(config.batch_size):
                initial_beam = CaptionData(sentence=[],
                                           memory=initial_memory[k],
                                           output=initial_output[k],
                                           score=1.0)
                partial_caption_data.append(TopN(config.beam_size))
                partial_caption_data[-1].push(initial_beam)
                complete_caption_data.append(TopN(config.beam_size))

            # Run beam search
            for idx in range(config.max_caption_length):
                partial_caption_data_lists = []
                for k in range(config.batch_size):
                    data = partial_caption_data[k].extract()
                    partial_caption_data_lists.append(data)
                    partial_caption_data[k].reset()
                num_steps = 1 if idx == 0 else config.beam_size
                for b in range(num_steps):
                    if idx == 0:
                        last_word = np.zeros((config.batch_size), np.int32)
                    else:
                        last_word = np.array([pcl[b].sentence[-1]
                                              for pcl in partial_caption_data_lists],
                                             np.int32)

                    last_memory = np.array([pcl[b].memory
                                            for pcl in partial_caption_data_lists],
                                           np.float32)
                    last_output = np.array([pcl[b].output
                                            for pcl in partial_caption_data_lists],
                                           np.float32)

                    memory, output, scores = sess.run(
                        [self.memory, self.output, self.probs],
                        feed_dict={self.contexts: contexts,
                                   self.last_word: last_word,
                                   self.last_memory: last_memory,
                                   self.last_output: last_output})

                    # Find the beam_size most probable next words
                    for k in range(config.batch_size):
                        caption_data = partial_caption_data_lists[k][b]
                        words_and_scores = list(enumerate(scores[k]))
                        words_and_scores.sort(key=lambda x: -x[1])
                        words_and_scores = words_and_scores[0:config.beam_size + 1]
                        # Append each of these words to the current partial caption
                        for w, s in words_and_scores:
                            sentence = caption_data.sentence + [w]
                            score = caption_data.score * s
                            beam = CaptionData(sentence,
                                               memory[k],
                                               output[k],
                                               score)
                            if vocabulary.words[w] == '.':
                                complete_caption_data[k].push(beam)
                            else:
                                partial_caption_data[k].push(beam)
            # lstm + emotion
            emotion = sess.run(self.emotion_output,
                feed_dict={self.contexts: contexts,
                           self.last_word: last_word,
                           self.last_memory: last_memory,
                           self.last_output: last_output,
                           self.images: images})

            emotion_result = emotion.argmax(axis=1)

            results = []
            for k in range(config.batch_size):
                if complete_caption_data[k].size() == 0:
                    complete_caption_data[k] = partial_caption_data[k]
                results.append(complete_caption_data[k].extract(sort=True))

            return results, emotion_result

    def save(self, sess):
        """ Save the model. """
        config = self.config
        with sess.as_default():
            data = {v.name: v.eval() for v in tf.global_variables()}
            save_path = os.path.join(config.save_dir, str(self.global_step.eval()))

            print((" Saving the model to %s..." % (save_path+".npy")))
            np.save(save_path, data)
            info_file = open(os.path.join(config.save_dir, "config.pickle"), "wb")
            config_ = copy.copy(config)
            config_.global_step = self.global_step.eval()
            pickle.dump(config_, info_file)
            info_file.close()
        print("Model saved.")

    def load(self, sess, model_file=None):
        """ Load the model. """
        config = self.config
        if model_file is not None:
            save_path = model_file
        else:
            info_path = os.path.join(config.save_dir, "config.pickle")
            info_file = open(info_path, "rb")
            config = pickle.load(info_file)
            global_step = config.global_step
            info_file.close()
            save_path = os.path.join(config.save_dir,
                                     str(global_step)+".npy")

        print("Loading the model from %s..." %save_path)
        data_dict = np.load(save_path, allow_pickle=True).item()
        count = 0
        for v in tqdm(tf.global_variables()):
            if v.name in data_dict.keys():
                sess.run(v.assign(data_dict[v.name]))
                count += 1
        print("%d tensors loaded." %count)

    def load_cnn(self, session, data_path, ignore_missing=True):
        """ Load a pretrained CNN model. """
        print("Loading the CNN from %s..." %data_path)
        data_dict = np.load(data_path).item()
        count = 0
        for op_name in tqdm(data_dict):
            with tf.variable_scope(op_name, reuse = True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                        count += 1
                    except ValueError:
                        pass
        print("%d tensors loaded." %count)

