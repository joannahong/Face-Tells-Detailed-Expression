
import numpy as np
import cv2
import heapq
import pandas as pd
from utils.coco.pycocoevalcap.classify import Classify
from data_generation import summary


class ImageLoader(object):
    def __init__(self):
        self.bgr = True
        self.scale_shape = np.array([224, 224], np.int32)
        self.crop_shape = np.array([224, 224], np.int32)
        #self.mean = np.load(mean_file).mean(1).mean(1)

    def load_image(self, image_file):
        """ Load and preprocess an image. """
        image = cv2.imread(image_file)

        if self.bgr:
            temp = image.swapaxes(0, 2)
            temp = temp[::-1]
            image = temp.swapaxes(0, 2)

        image = cv2.resize(image, (self.scale_shape[0], self.scale_shape[1]))
        offset = (self.scale_shape - self.crop_shape) / 2
        offset = offset.astype(np.int32)
        image = image[offset[0]:offset[0]+self.crop_shape[0],
                      offset[1]:offset[1]+self.crop_shape[1]]
        #image = image - self.mean
        return image

    def load_images(self, image_files):
        """ Load and preprocess a list of images. """
        images = []
        for image_file in image_files:
            #
            # server locataion for MMI
            server_dir = '/mnt/joannahong'
            image_file = server_dir + image_file[1:]

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
        self.emotion = '/home/joannahong/PycharmProjects/FES-master/Emotion.csv'
        self.actionunit = '/home/joannahong/PycharmProjects/FES-master/ActionUnit.csv'
        self.intensity = '/home/joannahong/PycharmProjects/FES-master/Intensity.csv'

        # self.emotion = '/home/joannahong/project/FES-master/Emotion_disfa.csv'
        # self.actionunit = '/home/joannahong/project/FES-master/ActionUnit_disfa.csv'
        # self.intensity = '/home/joannahong/project/FES-master/Intensity.csv'

        self.classify = Classify(self.emotion, self.actionunit, self.intensity)

    def load_emotions(self, emotion_files):
        onehot_emo = self.classify.one_hot(self.classify.emotion)
        emotions = []
        for emotion_file in emotion_files:
            emotion = open(emotion_file, "r")
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
            # server locataion
            server_dir = '/mnt/joannahong'
            data = pd.read_csv(server_dir+data_file[1:])
            # data = pd.read_csv(data_file[:])
            emotion = (data['Emotion'].tolist())[0]
            emotion = onehot_emo[emotion-1]
            emotions.append(emotion)
        return emotions


if __name__ == '__main__':
    actionunit = '/home/joannahong/PycharmProjects/FES-master/ActionUnit.csv'
    emotion = '/home/joannahong/PycharmProjects/FES-master/Emotion.csv'
    intensity = '/home/joannahong/PycharmProjects/FES-master/Intensity.csv'
    eval2 = '/home/joannahong/PycharmProjects/FES-master/val/results_best/results.csv'
    dir2 = '/home/joannahong/PycharmProjects/FES-master/val/annotation.csv'

    emotion_file = ['/home/joannahong/PycharmProjects/FES-master/CK+/Emotion/S129/011/S129_011_00000018_emotion.txt',
                    '/home/joannahong/PycharmProjects/FES-master/CK+/Emotion/S050/006/S050_006_00000023_emotion.txt']
    emo = EmotionLoader()
    print(emo.load_emotions(emotion_file))
    #
    # classify = Classify(emotion, actionunit, intensity)
    # onehot_emo = classify.one_hot(classify.emotion)
    # # onehot_au = classify.one_hot(classify.actionunit)
    # # onehot_int = classify.one_hot(classify.intensity)
    # print(onehot_emo)
    # print(classify.emotion)

