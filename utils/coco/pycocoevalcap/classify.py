import os
import pandas as pd
import numpy as np
import sklearn.preprocessing
from string import punctuation
from data_generation import summary



class Classify(object):
    def __init__(self, emo_dir, au_dir, int_dir):
        self.emo_dir = emo_dir
        self.emotion = pd.read_csv(emo_dir)
        self.intensity = pd.read_csv(int_dir)
        self.actionunit = pd.read_csv(au_dir)

    def emotionCheck(self, emo_val_data, emo_gt_data):
        print ('computing emotion classification score...')

        num = 0
        num_corr = 0
        for i in range(len(emo_val_data)):
            emo_vals = emo_val_data[i]+1
            emo_val = self.one_hot(emo_vals)
            emo_gt = emo_gt_data[i]
            num_corr += np.sum(np.asarray(emo_val)*emo_gt)
            for j in range(len(emo_val_data[i])):
                num += 1

        print('{} correct emotion(s) out of {} emotions in emotion classification-- {:.5f} %'.format(num_corr, num,
                                                                                                (num_corr/float(num))))
        return num_corr/float(num)

    def emogenCheck(self, val_data, gt_data):
        print ('computing text classification score...')

        # eval2 = '/Users/joannahong/PycharmProjects/FES-master/val/results/results.csv'
        # dir2 = '/Users/joannahong/PycharmProjects/FES-master/val/annotation.csv'
        ann = pd.read_csv(gt_data)
        evalann = pd.read_csv(val_data)
        caption_ann = ann.iloc[:, 1:2]
        caption_evalann = evalann.iloc[:, 1:2]

        corr_emo = 0
        corr_gen = 0
        corr_int = 0
        for i in range(len(caption_ann)):
            # print(i)
            # print(caption_ann.iloc[i][0])
            # print(caption_evalann.iloc[i][0])
            cap_ann = self.split(caption_ann.iloc[i][0])
            cap_eval = self.split(caption_evalann.iloc[i][0])

            if len(cap_eval) < 4:
                continue

            # # text v1
            # if cap_eval[1] == cap_ann[1]:
            #     corr_emo += 1
            # else:
            #     col_num = list(self.emotion.columns[self.emotion.isin([cap_eval[1]]).any()])
            #     if len(col_num) == 0:
            #         col_num = list(self.emotion.columns[self.emotion.isin([cap_eval[1].capitalize()]).any()])
            #     col_num = int(col_num[0])
            #     siblings = self.emotion[str(col_num)].values
            #     siblings = siblings[~pd.isnull(siblings)]
            #     if cap_ann[1] in siblings or cap_ann[1].capitalize() in siblings:
            #         corr_emo += 1
            #
            # if cap_ann[2] == 'woman':
            #     check = 'female'
            # if cap_ann[2] == 'man':
            #     check = 'male'
            #
            # if cap_eval[2] == cap_ann[2] or cap_eval[2] == check:
            #     corr_gen += 1
            #
            # text v2
            if cap_eval[3] == cap_ann[3]:
                corr_emo += 1
            else:
                col_num = list(self.emotion.columns[self.emotion.isin([cap_eval[3]]).any()])
                if len(col_num) == 0:
                    col_num = list(self.emotion.columns[self.emotion.isin([cap_eval[3].capitalize()]).any()])
                col_num = int(col_num[0])
                siblings = self.emotion[str(col_num)].values
                siblings = siblings[~pd.isnull(siblings)]
                if cap_ann[3] in siblings or cap_ann[3].capitalize() in siblings:
                    corr_emo += 1

            if cap_ann[1] == 'woman':
                check = 'female'
            if cap_ann[1] == 'man':
                check = 'male'

            if cap_eval[1] == cap_ann[1] or cap_eval[1] == check:
                corr_gen += 1

        print('{} correct emotion(s) out of {} emotions -- {:.2f}'.format(corr_emo, len(caption_ann),
                                                                           corr_emo/float(len(caption_ann))))
        print('{} correct gender(s) out of {} genders -- {:.2f}'.format(corr_gen, len(caption_ann),
                                                                         corr_gen/float(len(caption_ann))))

    def auintCheck(self, val_data, gt_data):
        return

    def split(self, sentence):
        '''

        :param sentence:
        :return:
        '''
        sentence = sentence.lower()
        sentence = ''.join(c for c in sentence if c not in punctuation)
        sentence = sentence.split()

        return sentence

    def one_hot(self, data):
        column_data = list(data)
        if 'Action Unit' in column_data:
            column_data.remove('Action Unit')
        column_data = map(int, column_data)
        column_data[:] = [x-1 for x in column_data]

        label_binarizer = sklearn.preprocessing.LabelBinarizer()
        if self.emo_dir[-9:] == 'disfa.csv':
            label_binarizer.fit(range(6))
        else:
            label_binarizer.fit(range(7))
        one_hot = label_binarizer.transform(column_data)

        return one_hot

if __name__ == '__main__':
    actionunit = '/home/joannahong/project/FES-master/ActionUnit.csv'
    emotion = '/home/joannahong/project/FES-master/Emotion.csv'
    intensity = '/home/joannahong/project/FES-master/Intensity.csv'
    eval2 = '/home/joannahong/project/FES-master/val/results/results.csv'
    dir2 = '/home/joannahong/project/FES-master/DISFA+/annotation/cross_val/1_test.csv'

    classify = Classify(emotion, actionunit, intensity)
    classify.emogenCheck(eval2, dir2)

    # ann = pd.read_csv(dir2)
    # evalann = pd.read_csv(eval2)
    # emo = pd.read_csv(emotion)
    # inten = pd.read_csv(intensity)
    #
    # caption_ann = ann.iloc[:,1:2]
    # caption_evalann = evalann.iloc[:,1:2]
    #
    # print(emo)
    # print(list(emo.columns[(emo == 'disgusted').iloc[0]]))
    # print(emo.columns[emo.isin(['Scornful']).any()])
    # print(emo.to_dict())
    # emo1 = np.asarray(emo[str(1)].values)
    # print(type(emo1))
    # emo1 = emo1[~pd.isnull(emo1)]
    # print(emo1)
    # print(inten.to_dict())
    #
    # corr_emo = 0
    # corr_gen = 0
    # corr_int = 0
    # for i in range(len(caption_ann)):
    #     print(i)
    #     print(caption_ann.iloc[i][0])
    #     print(caption_evalann.iloc[i][0])
    #     cap_ann = classify.split(caption_ann.iloc[i][0])
    #     cap_eval = classify.split(caption_evalann.iloc[i][0])
    #
    #     if cap_eval[1] == cap_ann[1]:
    #         corr_emo += 1
    #     else:
    #         col_num = list(emo.columns[emo.isin([cap_eval[1]]).any()])
    #         if len(col_num) == 0:
    #             col_num = list(emo.columns[emo.isin([cap_eval[1].capitalize()]).any()])
    #         col_num = int(col_num[0])
    #
    #         siblings = emo[str(col_num)].values
    #         siblings = siblings[~pd.isnull(siblings)]
    #         if cap_ann[1] in siblings or cap_ann[1].capitalize() in siblings:
    #             corr_emo += 1
    #
    #     if cap_eval[2] == cap_ann[2]:
    #         corr_gen += 1
    #
    #
    # print('{} correct emotion(s) out of {} emotions'.format(corr_emo, len(caption_ann)))
    # print('{} correct gender(s) out of {} genders'.format(corr_gen, len(caption_ann)))
    #
    #
    #

    #
    # classify = Classify(emotion, actionunit, intensity)
    # onehot_emo = classify.one_hot(classify.emotion)
    # # onehot_au = classify.one_hot(classify.actionunit)
    # # onehot_int = classify.one_hot(classify.intensity)
    # print(onehot_emo)
    # print(classify.emotion)
    # print(onehot_au)
    # print(onehot_int)
    # ann = pd.read_csv(dir2)
    # print(os.path.dirname(__file__))
    # print(list(ann.loc[0,['gender', 'emotion', 'action_unit']]))
    # df = summary(['/home/joannahong/PycharmProjects/FES-master/CK+/FACS/S134/gender.txt',
    #               '/home/joannahong/PycharmProjects/FES-master/CK+/Emotion/S134/003/S134_003_00000011_emotion.txt',
    #               '/home/joannahong/PycharmProjects/FES-master/CK+/FACS/S134/003/S134_003_00000011_facs.txt'])
    # print(df)
    # df1 = list(df.loc[4, ['Gender', 'Emotion', 'ActionUnit', 'Intensity']])
    # print(df1)
    # print([int(df1[0])])
    # print(onehot_emo[int(df1[1])-1])
    # print(onehot_au[int(df1[2])-1])
    # print(onehot_int[int(df1[3])-1])
    #
