
import numpy as np
import cv2
import heapq
import pandas as pd
from utils.coco.pycocoevalcap.classify import Classify
from config import Config

from data_generation import summary


class ImageLoader(object):
    def __init__(self):
        self.config = Config()
        self.bgr = True
        self.scale_shape = np.array([224, 224], np.int32)
        self.crop_shape = np.array([224, 224], np.int32)
        #self.mean = np.load(mean_file).mean(1).mean(1)

    def load_image(self, image_file):
        """ Load and preprocess an image. """
        image = cv2.imread(self.config.img_dir+image_file[1:])

        if self.bgr:
            temp = image.swapaxes(0, 2)
            temp = temp[::-1]
            image = temp.swapaxes(0, 2)

        image = cv2.resize(image, (self.scale_shape[0], self.scale_shape[1]))

        return image

    def load_images(self, image_files):
        """ Load and preprocess a list of images. """
        images = []
        for image_file in image_files:
            images.append(self.load_image(image_file))
        images = np.array(images, np.float32)

        return images

class CaptionData(object):
    def __init__(self, sentence, memory, output, score):

       self.sentence = sentence
       self.memory = memory
       self.output = output
       self.score = score

    def __cmp__(self, other):
        assert isinstance(other, CaptionData)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1

    def __lt__(self, other):
        assert isinstance(other, CaptionData)
        return self.score < other.score

    def __eq__(self, other):
        assert isinstance(other, CaptionData)
        return self.score == other.score

class TopN(object):
    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        self._data = []

class EmotionLoader(object):
    def __init__(self):
        self.config = Config()
        self.classify = Classify(self.config.emotion_file, self.config.action_unit_file, self.config.intensity_file)

    def load_emotions(self, emotion_files):
        onehot_emo = self.classify.one_hot(self.classify.emotion)
        emotions = []
        for emotion_file in emotion_files:
            emotion = open(self.config.img_dir+emotion_file[1:], "r")
            emotion = emotion.read()
            emotion = np.float(emotion[:-5])
            emotion = np.int(emotion)
            emotion = onehot_emo[emotion-1]
            emotion = emotion.tolist()
            emotions.append(emotion)
        return emotions

    def load_data(self, data_files):
        onehot_emo = self.classify.one_hot(self.classify.emotion)
        emotions = []
        for data_file in data_files:

            if self.config.data == 'CK+' or not self.config.server:
                data = pd.read_csv(data_file[:])
            else:
                server_dir = self.config.img_dir
                data = pd.read_csv(server_dir+data_file[1:])

            emotion = (data['Emotion'].tolist())[0]
            emotion = onehot_emo[emotion-1]
            emotions.append(emotion)
        return emotions


