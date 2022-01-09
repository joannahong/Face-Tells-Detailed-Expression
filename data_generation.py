#-*-coding: utf-8-*-
from text_generation import *
from config import Config

import os
import json
import csv
import shutil
import random
from xml.etree.ElementTree import parse
import numpy as np
import pandas as pd
from collections import OrderedDict


def extractImdir(data_dir):
    img_dir_aligned_full = []
    img_dir_flip_full = []
    img_dir_aligned = []
    img_dir_flip = []

    dir_image = os.path.join(data_dir, 'cohn-kanade-images')
    files = sorted(os.listdir(dir_image))
    for name in files:
        dir_image_1 = os.path.join(dir_image, name)
        image_files = sorted(os.listdir(dir_image_1))

        for child_name in image_files:
            dir_image_2 = os.path.join(dir_image_1, child_name)

            if os.path.isdir(dir_image_2):
                image_files_2 = sorted(os.listdir(dir_image_2))
                dir_image_aligned = os.path.join(dir_image_2, image_files_2[-2])
                dir_image_flip = os.path.join(dir_image_2, image_files_2[-1])

                img_dir_aligned_full.append(dir_image_aligned)
                img_dir_flip_full.append(dir_image_flip)

    # _, emo_dir, _ = extractAUdir(data_dir)
    # for i in range(len(img_dir_aligned_full)):
    #     if any(img_dir_aligned_full[i][-29:-12] in img_string for img_string in emo_dir):
    #         img_dir_aligned.append(img_dir_aligned_full[i])
    #         img_dir_flip.append(img_dir_flip_full[i])

    return img_dir_aligned_full, img_dir_flip_full


def extractImdir_rough(data_dir):
    dir_full = []
    dir_image = os.path.join(data_dir, 'cohn-kanade-images')

    files = sorted(os.listdir(dir_image))
    for name in files:
        dir_image_1 = os.path.join(dir_image, name)
        image_files = sorted(os.listdir(dir_image_1))

        for child_name in image_files:
            dir_image_2 = os.path.join(dir_image_1, child_name)

            if os.path.isdir(dir_image_2):
                image_files_2 = sorted(os.listdir(dir_image_2))

                for img in image_files_2:
                    dir_image_full = os.path.join(dir_image_2, img)
                    if dir_image_full[-3:] == 'png' and not dir_image_full[-11:-4] == 'aligned' and not dir_image_full[
                                                                                                        -11:-4] == 'flipped':
                        dir_full.append(dir_image_full)

    return dir_full


def extractAUdir(data_dir):
    '''extract Action Unit directory (list)'''

    Genderdir = []
    Emotiondir = []
    AUdir = []

    ### mmi dataset has no emotion ###
    if 'mmi' in data_dir:
        files = sorted(os.listdir(data_dir))
        files.sort(key=int)  # sort
        for name in files:
            fn = os.path.join(data_dir, name)
            aucs = filter(lambda x: 'aucs' in x, os.listdir(fn))[0]
            fn_aucs = os.path.join(fn, aucs)
            AUdir.append(fn_aucs)

    elif 'CK+' in data_dir:
        emo_fn = os.path.join(data_dir, 'Emotion')
        au_fn = os.path.join(data_dir, 'FACS')
        files = sorted(os.listdir(emo_fn))

        for name in files:
            emo_fn_full = os.path.join(emo_fn, name)
            au_fn_full = os.path.join(au_fn, name)
            gd_fn_full = os.path.join(au_fn_full, 'gender.txt')

            sess = sorted(os.listdir(au_fn_full))
            for aucs_dir in sess:
                if not 'gender' in aucs_dir:
                    au_fn_aucs_dir = os.path.join(au_fn_full, aucs_dir)
                    emo_fn_aucs_dir = os.path.join(emo_fn_full, aucs_dir)

                    # if os.listdir(emo_fn_aucs_dir):
                    Genderdir.append(gd_fn_full)

                    aucs = os.listdir(au_fn_aucs_dir)[0]
                    fn_aucs = os.path.join(au_fn_aucs_dir, aucs)
                    AUdir.append(fn_aucs)

                    if not os.path.exists(emo_fn_aucs_dir):
                        fn_emos = '.'
                    elif not os.listdir(emo_fn_aucs_dir):
                        fn_emos = '.'
                    else:
                        emos = os.listdir(emo_fn_aucs_dir)[0]
                        fn_emos = os.path.join(emo_fn_aucs_dir, emos)
                    Emotiondir.append(fn_emos)


    return Genderdir, Emotiondir, AUdir


def extractAU(data_dir):
    ''' extract Action Unit data with intensity '''
    type_dir = data_dir[-3:]

    if type_dir == 'xml':
        tree = parse(data_dir)
        root = tree.getroot()
        AU = pd.DataFrame()

        for ActionUnit in root.findall('ActionUnit'):
            numAU = ActionUnit.attrib

            for marker in ActionUnit:
                dict = marker.attrib
                dict.update(numAU)
                dict['Intensity'] = dict.pop('Type')
                dict['ActionUnit'] = dict.pop('Number')
                AU = AU.append(dict, ignore_index=True)

    elif type_dir == 'txt':
        AU = pd.read_csv(data_dir, delim_whitespace=True, names=["ActionUnit", "Intensity"])

    else:
        return None

    return AU


def extractInfo(data_dir, name='Gender'):
    '''gender or emotion information extraction'''
    type_dir = data_dir[-3:]

    if type_dir == 'txt':
        Emo = pd.read_csv(data_dir, delim_whitespace=True, names=[name])

    return Emo


def shuffleDataset(dir):
    img_aligned, img_flip = extractImdir(dir)
    gen, emo, au = extractAUdir(dir)

    shuffleAU = list(zip(img_aligned, img_flip, gen, emo, au))
    random.shuffle(shuffleAU)
    img_aligned, img_flip, gen, emo, au = zip(*shuffleAU)

    return [img_aligned, img_flip, gen, emo, au]


def creatDataset(data, num, config):
    img_aligned, img_flip, gen, emo, au = data
    # img_aligned, img_flip = extractImdir(dir)
    # gen, emo, au = extractAUdir(dir)
    anns, imgs, genders, emos, aus = [], [], [], [], []
    ann = OrderedDict()

    for i in range(len(emo)-50):  # last 100 for validation set
        df = summary([gen[i], emo[i], au[i]])
        text = generateText(df, numtext=num*2, orderS=config.orderS, orderG=config.orderG)

        for j in range(num):
            anns.append(text[j])
            imgs.append(img_aligned[i])
            genders.append(gen[i])
            emos.append(emo[i])
            aus.append(au[i])

        for k in range(num):
            anns.append(text[k+num])
            imgs.append(img_flip[i])
            genders.append(gen[i])
            emos.append(emo[i])
            aus.append(au[i])

    ann['caption'] = anns
    ann['image_file'] = imgs
    ann['gender'] = genders
    ann['emotion'] = emos
    ann['action_unit'] = aus

    ann = pd.DataFrame.from_dict(ann)

    data_dir = dir + 'annotation.csv'
    if os.path.exists(data_dir):
        os.remove(data_dir)
    ann.to_csv('annotation.csv')

    return ann


def creatDataset_test(data, num, config):
    img_aligned, img_flip, gen, emo, au = data
    # img_aligned, img_flip = extractImdir(dir)
    # gen, emo, au = extractAUdir(dir)
    anns, imgs, genders, emos, aus = [], [], [], [], []
    ann = OrderedDict()

    for q in range(50):  # last 100 for validation set
        i=len(emo)-50+q
        df = summary([gen[i], emo[i], au[i]])
        text = generateText(df, numtext=num*2, orderS=config.orderS, orderG=config.orderG)

        for j in range(num):
            anns.append(text[j])
            imgs.append(img_aligned[i])
            genders.append(gen[i])
            emos.append(emo[i])
            aus.append(au[i])

        for k in range(num):
            anns.append(text[k+num])
            imgs.append(img_flip[i])
            genders.append(gen[i])
            emos.append(emo[i])
            aus.append(au[i])

    ann['caption'] = anns
    ann['image_file'] = imgs
    ann['gender'] = genders
    ann['emotion'] = emos
    ann['action_unit'] = aus

    ann = pd.DataFrame.from_dict(ann)

    data_dir = dir + 'annotation_test.csv'
    if os.path.exists(data_dir):
        os.remove(data_dir)
    ann.to_csv('annotation_test.csv')

    return ann


def summary(data_dirs):
    '''
    :param data_dirs: [gender directory, emotion directory, action unit directory]
    :return:
    '''
    [gen, emo, au] = data_dirs
    dfgen = extractInfo(gen, 'Gender')
    if emo == '.':
        data  = {'Emotion':['0']}
        dfemo = pd.DataFrame(data)
    else:
        dfemo = extractInfo(emo, 'Emotion')

    dfau = extractAU(au)

    df = pd.concat([dfgen, dfemo], axis=1)
    df = pd.concat([df, dfau], axis=1)
    df['Emotion'] = df['Emotion'].fillna(df['Emotion'][0])
    df['Gender'] = df['Gender'].fillna(df['Gender'][0])

    # df = df.fillna("")  # for good-looking

    return df


if __name__ == '__main__':
    dir = './CK+'
    # dir2 = './mmi-facial-expression-database/Sessions'
    # dir3 = './mmi-facial-expression-database2/

    img_aligned, img_flip = extractImdir(dir)
    gen, emo, au = extractAUdir(dir)

    # data = shuffleDataset(dir)
    # print(data)


    # a = pd.read_csv('directory.csv')
    # print(a)
    # print(list(a.iloc[0][0:2]))
    # print(list(a.iloc[0][2:5]))

    full = []
    for i in range(len(emo)):
        print('Image directory (aligned) : {}'.format(img_aligned[i]))
        print('Image directory (flipped) : {}'.format(img_flip[i]))
        print('Gender directory: {}'.format(gen[i]))
        print('Emotion directory: {}'.format(emo[i]))
        print('ActionUnit directory: {}'.format(au[i]))

        data_dir = [gen[i], emo[i], au[i]]
        df = summary(data_dir)
        print(np.asarray(df['ActionUnit'])-1)
    #     #print(set(df['Gender']), set(df['Emotion']), df['ActionUnit'], df['Intensity'])
    #
    #     full.append(df)
    #
    # full_df = pd.concat(full)

    # print(full_df)
    # print(set(full_df['Gender']))
    # print(set(full_df['Emotion']))
    # print(set(full_df['ActionUnit']))
    # print(set(full_df['Intensity']))

    ################ Example ################
    #   Gender Emotion  ActionUnit  Intensity
    # 0      0       4         1.0        4.0
    # 1                        2.0        4.0
    # 2                        4.0        2.0
    # 3                        5.0        4.0
    # 4                       20.0        3.0
    # 5                       25.0        1.0
