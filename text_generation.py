import os
import pandas as pd
import random
import csv
from data_generation import *
# from MMI.dataset import summary

# CK+ dataset
# 0 = neutral, 1 = anger, 2 = contempt, 3 = disgust, 4 = fear, 5 = happy, 6 = sadness, 7 = surprise

def textParam(dataframe):
    # dfEmotion = pd.read_csv('Emotion.csv')
    # dfIntensity = pd.read_csv('Intensity.csv')
    # dfActionUnit = pd.read_csv('ActionUnit.csv')

    dfEmotion = pd.read_csv('Emotion_mmi.csv')
    dfIntensity = pd.read_csv('Intensity.csv')
    dfActionUnit = pd.read_csv('ActionUnit_mmi.csv')

    # gender
    if int((set(dataframe['Gender'])).pop()) == 0:
       Gender = 'man'

    elif int((set(dataframe['Gender'])).pop()) == 1:
       Gender = 'woman'

    # emotion
    numEmotion = int((set(dataframe['Emotion'])).pop())
    listEmotion = list(dfEmotion[str(numEmotion)])
    listEmotion = [Emo for Emo in listEmotion if str(Emo) != 'nan']  # remove nan
    Emotion = random.choice(listEmotion)

    # action unit with intensity
    Intensity = []
    AUnoun = []
    AUadj = []
    AUI = dataframe[['ActionUnit', 'Intensity']]
    numau = []  ### for constraint
    for i in range(len(AUI)):
        aui_comb = dict(AUI.iloc[i])
        AUAU = aui_comb['ActionUnit']

        if not str(AUAU)[-1] == '0':
            numActionUnit = aui_comb['ActionUnit']
        else:
            numActionUnit = int(aui_comb['ActionUnit'])
        numIntensity = int(round(aui_comb['Intensity']))
        if numIntensity < 0:
            numIntensity = 0

        numau.append(numActionUnit)  ### for constraint

        # action unit with noun and adjective(2 types)
        AUnoun.append(dfActionUnit.iloc[2][str(numActionUnit)])
        listAUadj = list(dfActionUnit.iloc[-2:][str(numActionUnit)])
        if type(listAUadj[1]) == float:
            AUadj.append(listAUadj[0])
        else:
            randAUadj = random.choice(listAUadj)
            AUadj.append(randAUadj)

        # intensity
        if numIntensity == 0:
            randIntensity = None
            Intensity.append(randIntensity)
        else:
            listIntensity = list(dfIntensity[str(numIntensity)])
            listIntensity = [Int for Int in listIntensity if str(Int) != 'nan']  # remove nan
            randIntensity = random.choice(listIntensity)
            Intensity.append(randIntensity)

    return Gender, Emotion, Intensity, AUnoun, AUadj, numau


def generateText2(dataframe, numtext, orderS=False, orderG=False, rand=False):
    '''
    :param param: [Gender, Emotion, Intensity, AUnoun, AUadj]
    :return: number of texts
    '''
    tlist = []
    for t in range(numtext):
        if rand == True:
            pass
        else:
            Gender, Emotion, Intensity, AUnoun, AUadj, numau = textParam(dataframe)

        if not orderS:
            # change order of AU
            shuffleAU = list(zip(Intensity, AUnoun, AUadj))
            random.shuffle(shuffleAU)
            Intensity, AUnoun, AUadj = zip(*shuffleAU)

        Emotion = Emotion.lower()

        if Gender == 'man':
            pronoun = 'he'
            pronoun2 = 'his'

        else:
            pronoun = 'she'
            pronoun2 = 'her'

        # text generation
        txt = 'A ' + Gender + ' is ' + Emotion + ' because '

        if not orderG:
            for i in range(len(AUnoun)):
                if random.random() > 0.5:
                    # if there is unknown intensity
                    if not Intensity[i]:
                        if len(AUnoun) == 1:
                            txt2 = pronoun + ' has ' + AUadj[i] + ' ' + AUnoun[i] + '.'
                        elif i == len(AUnoun) - 1 and len(AUnoun) != 1:
                            txt2 = 'and ' + pronoun + ' has ' + AUadj[i] + ' ' + AUnoun[i] + '.'
                        else:
                            txt2 = pronoun + ' has' + AUadj[i] + ' ' + AUnoun[i] + ', '

                    else:
                        if len(AUnoun) == 1:
                            txt2 = pronoun + ' has ' + Intensity[i] + ' ' + AUadj[i] + ' ' + AUnoun[i] + '.'
                        elif i == len(AUnoun) - 1 and len(AUnoun) != 1:
                            txt2 = 'and ' + pronoun + ' has ' + Intensity[i] + ' ' + AUadj[i] + ' ' + AUnoun[i] + '.'
                        else:
                            txt2 = pronoun + ' has ' + Intensity[i] + ' ' + AUadj[i] + ' ' + AUnoun[i] + ', '
                    txt = txt + txt2

                else:
                    # if there is unknown intensity
                    if AUnoun[i][-1] == 's':
                        beverb = ' are '
                    else:
                        beverb = ' is '

                    if not Intensity[i]:
                        if len(AUnoun) == 1:
                            txt2 = pronoun2 + ' ' + AUnoun[i] + beverb + AUadj[i] + '.'
                        elif i == len(AUnoun) - 1 and len(AUnoun) != 1:
                            txt2 = 'and ' + pronoun2 + ' ' + AUnoun[i] + beverb + AUadj[i] + '.'
                        else:
                            txt2 = pronoun2 + ' ' + AUnoun[i] + beverb + AUadj[i] + ', '

                    else:
                        if random.random() > 0.5 or Intensity[i] == 'somewhat':
                            if len(AUnoun) == 1:
                                txt2 = pronoun2 + ' ' + AUnoun[i] + beverb + Intensity[i] + ' ' + AUadj[i] + '.'
                            elif i == len(AUnoun) - 1 and len(AUnoun) != 1:
                                txt2 = 'and ' + pronoun2 + ' ' + AUnoun[i] + beverb + Intensity[i] + ' ' + AUadj[i] + '.'
                            else:
                                txt2 = pronoun2 + ' ' + AUnoun[i] + beverb + Intensity[i] + ' ' + AUadj[i] + ', '
                        else:
                            if len(AUnoun) == 1:
                                txt2 = pronoun2 + ' ' + AUnoun[i] + beverb + AUadj[i] + ' ' + Intensity[i] + '.'
                            elif i == len(AUnoun) - 1 and len(AUnoun) != 1:
                                txt2 = 'and ' + pronoun2 + ' ' + AUnoun[i] + beverb + AUadj[i] + ' ' + Intensity[i] + '.'
                            else:
                                txt2 = pronoun2 + ' ' + AUnoun[i] + beverb + AUadj[i] + ' ' + Intensity[i] + ', '

                    txt = txt + txt2

        else:
            if random.random() > 0.5:
                for i in range(len(AUnoun)):
                    # if there is unknown intensity
                    if not Intensity[i]:
                        if len(AUnoun) == 1:
                            txt2 = pronoun + ' has ' + AUadj[i] + ' ' + AUnoun[i] + '.'
                        elif i == len(AUnoun) - 1 and len(AUnoun) != 1:
                            txt2 = 'and ' + pronoun + ' has ' + AUadj[i] + ' ' + AUnoun[i] + '.'
                        else:
                            txt2 = pronoun + ' has' + AUadj[i] + ' ' + AUnoun[i] + ', '

                    else:
                        if len(AUnoun) == 1:
                            txt2 = pronoun + ' has ' + Intensity[i] + ' ' + AUadj[i] + ' ' + AUnoun[i] + '.'
                        elif i == len(AUnoun) - 1 and len(AUnoun) != 1:
                            txt2 = 'and ' + pronoun + ' has ' + Intensity[i] + ' ' + AUadj[i] + ' ' + AUnoun[
                                i] + '.'
                        else:
                            txt2 = pronoun + ' has ' + Intensity[i] + ' ' + AUadj[i] + ' ' + AUnoun[i] + ', '
                    txt = txt + txt2

                else:
                    # if there is unknown intensity
                    if AUnoun[i][-1] == 's':
                        beverb = ' is '
                    else:
                        beverb = ' are '

                    if not Intensity[i]:
                        if len(AUnoun) == 1:
                            txt2 = pronoun2 + ' ' + AUnoun[i] + beverb + AUadj[i] + '.'
                        elif i == len(AUnoun) - 1 and len(AUnoun) != 1:
                            txt2 = 'and ' + pronoun2 + ' ' + AUnoun[i] + beverb + AUadj[i] + '.'
                        else:
                            txt2 = pronoun2 + ' ' + AUnoun[i] + beverb + AUadj[i] + ', '

                    else:
                        if random.random > 0.5:
                            if len(AUnoun) == 1:
                                txt2 = pronoun2 + ' ' + AUnoun[i] + beverb + Intensity[i] + ' ' + AUadj[i] + '.'
                            elif i == len(AUnoun) - 1 and len(AUnoun) != 1:
                                txt2 = 'and ' + pronoun2 + ' ' + AUnoun[i] + beverb + Intensity[i] + ' ' + AUadj[i] + '.'
                            else:
                                txt2 = pronoun2 + ' ' + AUnoun[i] + beverb + Intensity[i] + ' ' + AUadj[i] + ', '
                        else:
                            if len(AUnoun) == 1:
                                txt2 = pronoun2 + ' ' + AUnoun[i] + beverb + AUadj[i] + ' ' + Intensity[i] + '.'
                            elif i == len(AUnoun) - 1 and len(AUnoun) != 1:
                                txt2 = 'and ' + pronoun2 + ' ' + AUnoun[i] + beverb + AUadj[i] + ' ' + Intensity[i] + '.'
                            else:
                                txt2 = pronoun2 + ' ' + AUnoun[i] + beverb + AUadj[i] + ' ' + Intensity[i] + ', '

                    txt = txt + txt2

        tlist.append(txt)

    return tlist, numau


def generateText(dataframe, numtext, orderS=False, orderG=False, rand=False):
    '''
    :param param: [Gender, Emotion, Intensity, AUnoun, AUadj]
    :return: number of texts
    '''
    preposition = [', subject with ', ' with ', ' subject with ']
    tlist = []
    for t in range(numtext):
        if rand == True:
            pass
        else:
            Gender, Emotion, Intensity, AUnoun, AUadj, numau = textParam(dataframe)

        if not orderS:
            # change order of AU
            shuffleAU = list(zip(Intensity, AUnoun, AUadj))
            random.shuffle(shuffleAU)
            Intensity, AUnoun, AUadj = zip(*shuffleAU)

        Emotion = Emotion.lower()

        # text generation
        if Emotion == 'neutral':
            prep = random.choice(preposition)

            txt = 'A ' + Gender + prep
        else:
            prep = random.choice(preposition)

            if Emotion[0] in ('a', 'e', 'i', 'o', 'u'):
                txt = 'An ' + Emotion + ' ' + Gender + prep
            else:
                txt = 'A ' + Emotion + ' ' + Gender + prep

        if not orderG:
            for i in range(len(AUnoun)):
                if random.random() > 0.5:
                    # if there is unknown intensity
                    if not Intensity[i]:
                        if len(AUnoun) == 1:
                            txt2 = AUnoun[i] + ' ' + AUadj[i] + '.'
                        elif len(AUnoun) == 2 and i == 0:
                            txt2 = AUnoun[i] + ' ' + AUadj[i] + ' '
                        elif i == len(AUnoun) - 1 and len(AUnoun) != 1:
                            txt2 = 'and ' + AUnoun[i] + ' ' + AUadj[i] + '.'
                        else:
                            txt2 = AUnoun[i] + ' ' + AUadj[i] + ', '

                    else:
                        if len(AUnoun) == 1:
                            txt2 = AUnoun[i] + ' ' + Intensity[i] + ' ' + AUadj[i] + '.'
                        elif len(AUnoun) == 2 and i == 0:
                            txt2 = AUnoun[i] + ' ' + Intensity[i] + ' ' + AUadj[i] + ' '
                        elif i == len(AUnoun) - 1 and len(AUnoun) != 1:
                            txt2 = 'and ' + AUnoun[i] + ' ' + Intensity[i] + ' ' + AUadj[i] + '.'
                        else:
                            txt2 = AUnoun[i] + ' ' + Intensity[i] + ' ' + AUadj[i] + ', '
                    txt = txt + txt2

                else:
                    # if there is unknown intensity
                    if not Intensity[i]:
                        if len(AUnoun) == 1:
                            txt2 = AUadj[i] + ' ' + AUnoun[i] + '.'
                        elif len(AUnoun) == 2 and i == 0:
                            txt2 = AUadj[i] + ' ' + AUnoun[i] + ' '
                        elif i == len(AUnoun) - 1 and len(AUnoun) != 1:
                            txt2 = 'and ' + AUadj[i] + ' ' + AUnoun[i] + '.'
                        else:
                            txt2 = AUadj[i] + ' ' + AUnoun[i] + ', '

                    else:
                        if len(AUnoun) == 1:
                            txt2 = Intensity[i] + ' ' + AUadj[i] + ' ' + AUnoun[i] + '.'
                        elif len(AUnoun) == 2 and i == 0:
                            txt2 = Intensity[i] + ' ' + AUadj[i] + ' ' + AUnoun[i] + ' '
                        elif i == len(AUnoun) - 1 and len(AUnoun) != 1:
                            txt2 = 'and ' + Intensity[i] + ' ' + AUadj[i] + ' ' + AUnoun[i] + '.'
                        else:
                            txt2 = Intensity[i] + ' ' + AUadj[i] + ' ' + AUnoun[i] + ', '

                    txt = txt + txt2

        else:
            if random.random() > 0.5:
                for i in range(len(AUnoun)):
                    # if there is unknown intensity
                    if not Intensity[i]:
                        if len(AUnoun) == 1:
                            txt2 = AUnoun[i] + ' ' + AUadj[i] + '.'
                        elif i == len(AUnoun) - 1 and len(AUnoun) != 1:
                            txt2 = 'and ' + AUnoun[i] + ' ' + AUadj[i] + '.'
                        else:
                            txt2 = AUnoun[i] + ' ' + AUadj[i] + ', '

                    else:
                        if len(AUnoun) == 1:
                            txt2 = AUnoun[i] + ' ' + Intensity[i] + ' ' + AUadj[i] + '.'
                        elif i == len(AUnoun) - 1 and len(AUnoun) != 1:
                            txt2 = 'and ' + AUnoun[i] + ' ' + Intensity[i] + ' ' + AUadj[i] + '.'
                        else:
                            txt2 = AUnoun[i] + ' ' + Intensity[i] + ' ' + AUadj[i] + ', '
                    txt = txt + txt2

            else:
                for i in range(len(AUnoun)):
                    # if there is unknown intensity
                    if not Intensity[i]:
                        if i == len(AUnoun) - 1 and len(AUnoun) != 1:
                            txt2 = 'and ' + AUadj[i] + ' ' + AUnoun[i] + '.'
                        else:
                            txt2 = AUadj[i] + ' ' + AUnoun[i] + ', '

                    else:
                        if i == len(AUnoun) - 1 and len(AUnoun) != 1:
                            txt2 = 'and ' + Intensity[i] + ' ' + AUadj[i] + ' ' + AUnoun[i] + '.'
                        else:
                            txt2 = Intensity[i] + ' ' + AUadj[i] + ' ' + AUnoun[i] + ', '

                    txt = txt + txt2

        tlist.append(txt)

    return tlist, numau



def generateText3(dataframe, numtext, orderS=False, orderG=False, rand=False):
    '''
    :param param: [Gender, Emotion, Intensity, AUnoun, AUadj]
    :return: number of texts
    '''
    tlist = []
    for t in range(numtext):
        if rand == True:
            pass
        else:
            Gender, Emotion, Intensity, AUnoun, AUadj, numau = textParam(dataframe)

        if not orderS:
            # change order of AU
            shuffleAU = list(zip(Intensity, AUnoun, AUadj))
            random.shuffle(shuffleAU)
            Intensity, AUnoun, AUadj = zip(*shuffleAU)

        Emotion = Emotion.lower()

        if Gender == 'man':
            gender_pro = "man's"
            pronoun = 'he'
            pronoun2 = 'his'

        else:
            gender_pro = "woman's"
            pronoun = 'she'
            pronoun2 = 'her'

        # text generation
        txt = 'A ' + gender_pro

        for i in range(len(AUnoun)):
            # if there is unknown intensity
            if AUnoun[i][-1] == 's':
                beverb = ' are '
            else:
                beverb = ' is '

            if not Intensity[i]:
                if len(AUnoun) == 1:
                    txt2 = ' ' + AUnoun[i] + beverb + AUadj[i] + '.'
                elif i == 0:
                    txt2 = ' ' + AUnoun[i] + beverb + AUadj[i] + ', '
                elif i == 1:
                    txt2 = pronoun2 + ' ' + AUnoun[i] + beverb + AUadj[i] + ', '
                elif i == len(AUnoun) - 1 and len(AUnoun) != 1:
                    txt2 = 'and ' + pronoun2 + ' ' + AUnoun[i] + beverb + AUadj[i] + '.'
                else:
                    txt2 = AUnoun[i] + beverb + AUadj[i] + ', '

            else:
                if random.random() > 0.5 or Intensity[i] == 'somewhat' or Intensity[i] == 'kind of':
                    if len(AUnoun) == 1:
                        txt2 = ' ' + AUnoun[i] + beverb + Intensity[i] + ' ' + AUadj[i] + '.'
                    elif i == 0:
                        txt2 = ' ' + AUnoun[i] + beverb + Intensity[i] + ' ' + AUadj[i] + ', '
                    elif len(AUnoun) == 2:
                        txt2 = 'and ' + pronoun2 + ' ' + AUnoun[i] + beverb + Intensity[i] + ' ' + AUadj[i] + '.'
                    elif i == len(AUnoun) - 1 and len(AUnoun) != 1:
                        txt2 = 'and ' + pronoun2 + ' ' + AUnoun[i] + beverb + Intensity[i] + ' ' + AUadj[i] + '.'
                    elif i == 1:
                        txt2 = pronoun2 + ' ' + AUnoun[i] + beverb + Intensity[i] + ' ' + AUadj[i] + ', '
                    else:
                        txt2 = pronoun2 + ' ' + AUnoun[i] + beverb + Intensity[i] + ' ' + AUadj[i] + ', '
                else:
                    if len(AUnoun) == 1:
                        txt2 = ' ' + AUnoun[i] + beverb + AUadj[i] + ' ' + Intensity[i] + '.'
                    elif i == 0:
                        txt2 = ' ' + AUnoun[i] + beverb + AUadj[i] + ' ' + Intensity[i] + ', '
                    elif len(AUnoun) == 2:
                        txt2 = 'and ' + pronoun2 + ' ' + AUnoun[i] + beverb + AUadj[i] + ' ' + Intensity[i] + '.'
                    elif i == len(AUnoun) - 1 and len(AUnoun) != 1:
                        txt2 = 'and ' + pronoun2 + ' ' + AUnoun[i] + beverb + AUadj[i] + ' ' + Intensity[i] + '.'
                    elif i ==1:
                        txt2 = pronoun2 + ' ' + AUnoun[i] + beverb + AUadj[i] + ' ' + Intensity[i] + ', '
                    else:
                        txt2 = pronoun2 + ' ' + AUnoun[i] + beverb + AUadj[i] + ' ' + Intensity[i] + ', '

            txt = txt + txt2



        tlist.append(txt)

    return tlist, numau



if __name__ == '__main__':
    dir = './CK+'
    # dir = './MMI/mmi-facial-expression-database/mmi-images'

    # data_directory = pd.read_csv('./DISFA+/annotation/cross_val2/data/0_test.csv')
    #
    # dictionary = {}
    # txt, img, data=[], [], []
    # for i in range(len(data_directory)):
    #     df = pd.read_csv(data_directory['data'][i])
    #     text, _ = generateText3(df, 1)
    #     for j in range(1):
    #         img.append(data_directory['image_file'][i])
    #         data.append(data_directory['data'][i])
    #         txt.append(text[j])
    # dictionary['caption'] = txt
    # dictionary['data'] = data
    # dictionary['image_file'] = img
    # dictionary = pd.DataFrame(dictionary)
    # dictionary.to_csv('./DISFA+/annotation/cross_val2/0_test.csv')
    # print('done.')
    #
    # data_directory = pd.read_csv('./DISFA+/annotation/cross_val2/data/1_test.csv')
    #
    # dictionary = {}
    # txt, img, data=[], [], []
    # for i in range(len(data_directory)):
    #     df = pd.read_csv(data_directory['data'][i])
    #     text, _ = generateText3(df, 1)
    #     for j in range(1):
    #         img.append(data_directory['image_file'][i])
    #         data.append(data_directory['data'][i])
    #         txt.append(text[j])
    # dictionary['caption'] = txt
    # dictionary['data'] = data
    # dictionary['image_file'] = img
    # dictionary = pd.DataFrame(dictionary)
    # dictionary.to_csv('./DISFA+/annotation/cross_val2/1_test.csv')
    # print('done.')

    #
    # data_directory = pd.read_csv('./DISFA+/annotation/cross_val2/data/2_test.csv')
    #
    # dictionary = {}
    # txt, img, data=[], [], []
    # for i in range(len(data_directory)):
    #     df = pd.read_csv(data_directory['data'][i])
    #     text, _ = generateText3(df, 1)
    #     for j in range(1):
    #         img.append(data_directory['image_file'][i])
    #         data.append(data_directory['data'][i])
    #         txt.append(text[j])
    # dictionary['caption'] = txt
    # dictionary['data'] = data
    # dictionary['image_file'] = img
    # dictionary = pd.DataFrame(dictionary)
    # dictionary.to_csv('./DISFA+/annotation/cross_val2/2_test.csv')
    # print('done.')
    #
    # data_directory = pd.read_csv('./DISFA+/annotation/cross_val2/data/3_test.csv')
    #
    # dictionary = {}
    # txt, img, data=[], [], []
    # for i in range(len(data_directory)):
    #     df = pd.read_csv(data_directory['data'][i])
    #     text, _ = generateText3(df, 1)
    #     for j in range(1):
    #         img.append(data_directory['image_file'][i])
    #         data.append(data_directory['data'][i])
    #         txt.append(text[j])
    # dictionary['caption'] = txt
    # dictionary['data'] = data
    # dictionary['image_file'] = img
    # dictionary = pd.DataFrame(dictionary)
    # dictionary.to_csv('./DISFA+/annotation/cross_val2/3_test.csv')
    # print('done.')
    #
    # data_directory = pd.read_csv('./DISFA+/annotation/cross_val2/data/4_test.csv')
    #
    # dictionary = {}
    # txt, img, data=[], [], []
    # for i in range(len(data_directory)):
    #     df = pd.read_csv(data_directory['data'][i])
    #     text, _ = generateText3(df, 1)
    #     for j in range(1):
    #         img.append(data_directory['image_file'][i])
    #         data.append(data_directory['data'][i])
    #         txt.append(text[j])
    # dictionary['caption'] = txt
    # dictionary['data'] = data
    # dictionary['image_file'] = img
    # dictionary = pd.DataFrame(dictionary)
    # dictionary.to_csv('./DISFA+/annotation/cross_val2/4_test.csv')
    # print('done.')
    #
    # data_directory = pd.read_csv('./DISFA+/annotation/cross_val2/data/5_test.csv')
    #
    # dictionary = {}
    # txt, img, data=[], [], []
    # for i in range(len(data_directory)):
    #     df = pd.read_csv(data_directory['data'][i])
    #     text, _ = generateText3(df, 1)
    #     for j in range(1):
    #         img.append(data_directory['image_file'][i])
    #         data.append(data_directory['data'][i])
    #         txt.append(text[j])
    # dictionary['caption'] = txt
    # dictionary['data'] = data
    # dictionary['image_file'] = img
    # dictionary = pd.DataFrame(dictionary)
    # dictionary.to_csv('./DISFA+/annotation/cross_val2/5_test.csv')
    # print('done.')
    #
    # data_directory = pd.read_csv('./DISFA+/annotation/cross_val2/data/6_test.csv')
    #
    # dictionary = {}
    # txt, img, data=[], [], []
    # for i in range(len(data_directory)):
    #     df = pd.read_csv(data_directory['data'][i])
    #     text, _ = generateText3(df, 1)
    #     for j in range(1):
    #         img.append(data_directory['image_file'][i])
    #         data.append(data_directory['data'][i])
    #         txt.append(text[j])
    # dictionary['caption'] = txt
    # dictionary['data'] = data
    # dictionary['image_file'] = img
    # dictionary = pd.DataFrame(dictionary)
    # dictionary.to_csv('./DISFA+/annotation/cross_val2/6_test.csv')
    # print('done.')
    #
    # data_directory = pd.read_csv('./DISFA+/annotation/cross_val2/data/7_test.csv')
    #
    # dictionary = {}
    # txt, img, data=[], [], []
    # for i in range(len(data_directory)):
    #     df = pd.read_csv(data_directory['data'][i])
    #     text, _ = generateText3(df, 1)
    #     for j in range(1):
    #         img.append(data_directory['image_file'][i])
    #         data.append(data_directory['data'][i])
    #         txt.append(text[j])
    # dictionary['caption'] = txt
    # dictionary['data'] = data
    # dictionary['image_file'] = img
    # dictionary = pd.DataFrame(dictionary)
    # dictionary.to_csv('./DISFA+/annotation/cross_val2/7_test.csv')
    # print('done.')
    #
    # data_directory = pd.read_csv('./DISFA+/annotation/cross_val2/data/8_test.csv')
    #
    # dictionary = {}
    # txt, img, data=[], [], []
    # for i in range(len(data_directory)):
    #     df = pd.read_csv(data_directory['data'][i])
    #     text, _ = generateText3(df, 1)
    #     for j in range(1):
    #         img.append(data_directory['image_file'][i])
    #         data.append(data_directory['data'][i])
    #         txt.append(text[j])
    # dictionary['caption'] = txt
    # dictionary['data'] = data
    # dictionary['image_file'] = img
    # dictionary = pd.DataFrame(dictionary)
    # dictionary.to_csv('./DISFA+/annotation/cross_val2/8_test.csv')
    # print('done.')

    #### text classification task

    # data_directory = pd.read_csv('directory.csv')
    # au_dir = '/home/joannahong/PycharmProjects/FES-master/cnn-text-classification-pytorch-master/au.csv'
    # text_dir = '/home/joannahong/PycharmProjects/FES-master/cnn-text-classification-pytorch-master/text.csv'
    #
    # # print(data_directory.index.get_values())  # index
    # print(data_directory.shape[0])
    # aus = []
    # texts = []
    # for i in range(data_directory.shape[0]):
    #     img_dir = data_directory.loc[i]['Image_aligned'], data_directory.loc[i]['Image_flipped']
    #     data_dir = data_directory.loc[i]['Gender'], data_directory.loc[i]['Emotion'], data_directory.loc[i]['Action Unit']
    #     df = summary(data_dir)
    #     # print(data_dir)
    #     # print(df)
    #     # print(img_dir)
    #     # print(data_dir)
    #     text, numau = generateText(df, numtext=100, orderS=False, orderG=False)
    #     for i in range(len(text)):
    #         aus.append([numau])
    #         texts.append([text[i]])
    #
    # with open(au_dir, "wb") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(aus)
    #
    # with open(text_dir, "wb") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(texts)
    #     #     #print(len(text[i].split()))

    # #### MMI
    # dir = './MMI/mmi-facial-expression-database/mmi-images'
    #
    # fds = os.listdir(dir)
    # fds.sort(key=int)
    # num_text = 1
    #
    # data, img, txt = [], [], []
    # a = {}
    # for name in fds:
    #     img_dir = os.path.join(dir, name)
    #     files = sorted(os.listdir(img_dir))
    #
    #
    #     for i in files:
    #         if 'png' in i:
    #             data_file = os.path.join(img_dir, i[:-3]+'csv')
    #             img_file = os.path.join(img_dir, i)
    #             # print(data_file)
    #             # print(img_file)
    #             df = pd.read_csv(data_file)
    #
    #             text, _ = generateText(df,num_text)
    #
    #             for i in range(num_text):
    #                 txt.append(text[i])
    #                 data.append(data_file)
    #                 img.append(img_file)
    #
    # # a['caption'] = txt
    # a['image_file'] = img
    # a['data'] = data
    #
    # df = pd.DataFrame.from_dict(a)
    # df.to_csv('./annotation_mmi/directory.csv')
    ################### MMI


    # print(len(data))
    # print(len(img))
    # print(len(txt))
    # print(df)
    #
    gen, emo, au = extractAUdir(dir)
    img_aligned, img_flip = extractImdir(dir)


    for i in range(len(emo)):
        print(len(emo))
        image_dir = [img_aligned[i], img_flip[i]]
        data_dir = [gen[i], emo[i], au[i]]
        print(image_dir)
        print(data_dir)
        df = summary(data_dir)
        print(df)
        # print(int((set(df['Gender'])).pop()))
        # print(int((set(df['Emotion'])).pop()))
        #
        # dfEmotion = pd.read_csv('Emotion.csv')
        # dfIntensity = pd.read_csv('Intensity.csv')
        # numEmotion = int((set(df['Emotion'])).pop())
        # listEmotion = list(dfEmotion[str(numEmotion)])
        # emo = random.choice(listEmotion)
        # print(listEmotion)
        # print(emo)
        # auin = df[['ActionUnit','Intensity']]
        # print(auin)
        # print(len(auin))
        # print(len(df))
        # aui_comb = dict(auin.iloc[0])
        # print(aui_comb['Intensity'])
        #
        #
        # gender, emotion, intensity, noun, adj = textParam(df)
        # print("{} {}, with {}".format(emotion, gender,intensity[0]))
        # # todo
        # # if None, do not print anything for intensity
        # print(gender)
        # print(emotion)
        # print(intensity)
        # print(noun)
        # print(adj)

        text = generateText(df, 3)
        print(text)
        print(text[0])
        print(text[1])
        print(text[2])
    #
    #
    #
    # full = []
    # for i in range(len(emo)):
    #     print('Gender directory: {}'.format(gen[i]))
    #     print('Emotion directory: {}'.format(emo[i]))
    #     print('ActionUnit directory: {}'.format(au[i]))
    #
    #     data_dir = [gen[i], emo[i], au[i]]
    #     df = summary(data_dir)
    #     print(df)
    #
    #     full.append(df)
    #
    # # full_df = pd.concat(full)
    #
    # # print(full_df)
    # # print(set(full_df['Gender']))
    # # print(set(full_df['Emotion']))
    # # print(set(full_df['ActionUnit']))
    # # print(set(full_df['Intensity']))
    #
    # ############### Example ################
    # #   Gender Emotion  ActionUnit  Intensity
    # # 0      0       4         1.0        4.0
    # # 1                        2.0        4.0
    # # 2                        4.0        2.0
    # # 3                        5.0        4.0
    # # 4                       20.0        3.0
    # # 5                       25.0        1.0

