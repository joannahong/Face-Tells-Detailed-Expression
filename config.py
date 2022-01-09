
class Config(object):
    """ Wrapper class for various (hyper)parameters. """
    def __init__(self):
        # about the model architecture
        self.cnn = 'vggface_emotion'  # 'vgg16' or 'vggface' or 'resnet50' or 'vgg16_emotion' or 'vgg16_modified'
        self.dim_embedding = 512
        self.num_lstm_units = 512
        self.num_initalize_layers = 2    # 1 or 2
        self.dim_initalize_layer = 512
        self.num_attend_layers = 2       # 1 or 2
        self.dim_attend_layer = 512
        self.num_decode_layers = 2       # 1 or 2
        self.dim_decode_layer = 1024

        # about the weight initialization and regularization
        self.fc_kernel_initializer_scale = 0.08
        self.fc_kernel_regularizer_scale = 1e-4
        self.fc_activity_regularizer_scale = 0
        self.conv_kernel_regularizer_scale = 1e-4
        self.conv_activity_regularizer_scale = 0
        self.fc_drop_rate = 0.5
        self.lstm_drop_rate = 0.3
        self.attention_loss_factor = 0.01

        # about the optimization
        self.num_epochs = 1000
        self.batch_size = 30
        self.optimizer = 'Adam'    # 'Adam', 'RMSProp', 'Momentum' or 'SGD'
        self.initial_learning_rate = 0.0001
        self.learning_rate_decay_factor = 1.0
        self.num_steps_per_decay = 100000
        self.clip_gradients = 5.0
        self.momentum = 0.0
        self.use_nesterov = True
        self.decay = 0.8
        self.centered = True
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-6

        self.dir = ''
        self.numtext = 5
        self.orderS = False
        self.orderG = False

        # about the training
        self.train_image_dir = './train/images/'
        self.train_caption_file = './train/captions_train2014.json'
        self.annotation_file_train = './train/annotation.csv'
        self.temp_data_file = './data.npy'

        # about the evaluation
        self.eval_image_dir = './val/images/'
        self.eval_caption_file = './val/captions_val2014.json'
        self.annotation_file_eval = './val/annotation.csv'
        self.eval_result_dir = './val/results/'
        self.eval_result_file = './val/results.json'
        self.save_eval_result_as_image = True

        # about the testing
        self.test_image_dir = './test/images/'
        self.test_result_dir = './test/results/'
        self.test_result_file = './test/results.csv'

        self.save_threshold = 10000
        self.save_period = 1000
        self.eval_period = 100

        # about data generation
        self.data = 'CK+'  # or 'mmi' 'CK+' 'DISFA'
        if self.data =='CK+':
            self.action_unit_file = './ActionUnit.csv'  # './ActionUnit_mmi.csv' or './ActionUnit.csv'
            self.intensity_file = './Intensity.csv'
            self.emotion_file = './Emotion.csv'  # './Emotion_mmi.csv' or './Emotion.csv'
        # elif self.data =='DISFA':
        #     self.action_unit_file = './ActionUnit_disfa.csv'  # './ActionUnit_mmi.csv' or './ActionUnit.csv'
        #     self.intensity_file = './Intensity.csv'
        #     self.emotion_file = './Emotion_disfa.csv'  # './Emotion_mmi.csv' or './Emotion.csv'
        # elif self.data =='mmi':
        #     self.action_unit_file = './ActionUnit_mmi.csv'  # './ActionUnit_mmi.csv' or './ActionUnit.csv'
        #     self.intensity_file = './Intensity.csv'
        #     self.emotion_file = './Emotion_mmi.csv'  # './Emotion_mmi.csv' or './Emotion.csv'

        # about the vocabulary
        self.vocabulary_size = 106  # 77 for disfa # 106 or 86 for ck # 5:91 , 108 mmi
        self.max_caption_length = 68  # 65 for disfa, 67, 68(3~), 69(9) for ck+ # 40 for 3, 60 for 1 mmi

        self.server = True
        self.is_train = True
        self.img_dir = '/mnt/hard1/joannahong/tmp/project/FES-master'  # parent directory of dataset

        cv_num = 3 ####### important

        # about the evaluation
        self.eval_image_dir = './val/images/'
        self.eval_caption_file = './val/captions_val2014.json'
        self.annotation_file_eval = './val/annotation.csv'
        self.eval_result_dir = './val/results{}/'.format(cv_num)
        self.eval_result_file = './val/results{}.json'.format(cv_num)
        self.save_eval_result_as_image = True

        ## cross validation for ck+
        self.save_dir = './models/ck+'
        self.summary_dir = './summary/ck+'
        self.temp_data_file = './data/ck+/data_cv{}.npy'.format(cv_num)
        self.vocabulary_file = './vocabulary/ck+/vocabulary_cv{}.csv'.format(cv_num)

        # about the training
        self.annotation_file_train = './cross_val/ck+/{}/annotation3.csv'.format(cv_num)
        # about the evaluation
        self.annotation_file_eval = './cross_val/ck+/{}/annotation_test3.csv'.format(cv_num)

        # ## cross validation for ck+
        # self.save_dir = '/mnt/hard1/joannahong/FES-master/vggonly/ck+/model/9/'
        # self.summary_dir = '/mnt/hard1/joannahong/FES-master/vggonly/ck+/summary/9/'
        # # self.save_dir = './models/0'
        # # self.summary_dir = './summary/0'
        # self.temp_data_file = './data_ck+/data_cv9.npy'
        # self.vocabulary_file = './vocabulary_ck+/vocabulary_cv9.csv'
        #
        # # about the training
        # self.annotation_file_train = './cross_val_ck+/9/annotation3.csv'
        # # about the evaluation
        # self.annotation_file_eval = './cross_val_ck+/9/annotation_test3.csv'

        ### vocabulary info
        # 0: 106
        # 1: 106
        # 2: 104
        # 3: 106
