import argparse
import numpy as np
# import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
# from scipy import stats
from utils import sequence_modeling
from utils import get_face_areas
from tqdm import tqdm
from utils.get_models import load_weights_EE, load_weights_LSTM, resnet18, cross_modal_attention
from utils import three_d_resnet_bi_lstm 
from utils import get_log_mel

import warnings
warnings.filterwarnings('ignore', category = FutureWarning)

parser = argparse.ArgumentParser(description="run")

parser.add_argument('--path_video', type=str, default='./test_video/', help='Path to all videos')
parser.add_argument('--path_save', type=str, default='report/', help='Path to save the report')
parser.add_argument('--conf_d', type=float, default=0.7, help='Elimination threshold for false face areas')
parser.add_argument('--path_FE_model', type=str, default="./models/EMOAffectNet_LSTM/EmoAffectnet/weights.h5",
                    help='Path to a model for feature extraction')
parser.add_argument('--path_LSTM_model', type=str, default="./models/EMOAffectNet_LSTM/LSTM/weights.h5",
                    help='Path to a model for emotion prediction')
                  

args = parser.parse_args()


EE_model = load_weights_EE(args.path_FE_model)
LSTM_model = load_weights_LSTM(args.path_LSTM_model)

model_video = three_d_resnet_bi_lstm.build_three_d_resnet_18((60, 88, 88, 3), 12, 'softmax', None,True, '3D')
model_video.build((None, 60, 88, 88, 3))
model_audio = resnet18(input_shape=(128,251,1), number_class=12)
model_av = cross_modal_attention(number_class=12)

def pred_one_video(path):
    all_time = time.time()
    label_phrase = ["DFA", "IEO", "IOM",
                    "ITH", "ITS", "IWL",
                    "IWW", "MTI", "TAI",
                    "TIE", "TSI", "WSI"]
                    
    start_time = time.time()
    detect = get_face_areas.VideoCamera(path_video=path, conf=args.conf_d)
    dict_face_areas, dict_lips_areas, total_frame = detect.get_frame()
    end_pre_video = time.time() - start_time
    
    name_frames = list(dict_face_areas.keys())
    face_areas = dict_face_areas.values()
    features = EE_model(np.stack(face_areas)).numpy()
    
    start_time = time.time()
    seq_features = sequence_modeling.get_sequence(name_frames, features)
    pred_emo = LSTM_model(np.stack(seq_features)).numpy()
    pred_sum = np.sum(pred_emo, axis=0)
    sum = np.sum(pred_sum)
    pred_emo = pred_sum/sum
    sort_pred = np.argsort(-pred_emo)
    if sort_pred[0]==3:
        pred_emo=sort_pred[1]
    else:
        pred_emo=sort_pred[0]
    end_emo = time.time() - start_time
    
    if pred_emo==0:
        weights_video = './models/VALENCE_VIDEO/{}/weights.h5'.format('NEU')
        weights_audio = './models/VALENCE_AUDIO/{}/weights.h5'.format('NEU')
        weights_av = './models/VALENCE_AV/{}/weights.h5'.format('NEU')
        p_emo = 'NEU'
    elif pred_emo==1:
        weights_video = './models/VALENCE_VIDEO/{}/weights.h5'.format('POS')
        weights_audio = './models/VALENCE_AUDIO/{}/weights.h5'.format('POS')
        weights_av = './models/VALENCE_AV/{}/weights.h5'.format('POS')
        p_emo = 'POS'
    else:
        weights_video = './models/VALENCE_VIDEO/{}/weights.h5'.format('NEG')
        weights_audio = './models/VALENCE_AUDIO/{}/weights.h5'.format('NEG')
        weights_av = './models/VALENCE_AV/{}/weights.h5'.format('NEG')
        p_emo = 'NEG'
        
    model_video.load_weights(weights_video)
    model_audio.load_weights(weights_audio)
    model_av.load_weights(weights_av)
    
    start_time = time.time()
    seg_features_audio = get_log_mel.get_log_mel(path) 
    prob_audio = model_audio(np.stack(seg_features_audio)).numpy()
    end_audio = time.time() - start_time
    
    start_time = time.time()
    name_frames_lips = list(dict_lips_areas.keys())    
    seq_features_lips = sequence_modeling.get_sequence(name_frames_lips, np.stack(dict_lips_areas.values()), win = 60, step = 30)
    prob_video = model_video(np.stack(seq_features_lips)).numpy()
    end_video = time.time() - start_time
    
    start_time = time.time()
    prob_audio = np.pad(prob_audio, ((0, 3-prob_audio.shape[0]), (0, 0)), 'mean')
    prob_audio = np.expand_dims(prob_audio, axis=0)
    prob_video = np.pad(prob_video, ((0, 3-prob_video.shape[0]), (0, 0)), 'mean')
    prob_video = np.expand_dims(prob_video, axis=0)
    prob_av = model_av([prob_audio, prob_video])
    end_time = time.time() - start_time
    all_time_end = time.time() - all_time
        
    print('Path: ', path)
    print('Predicted emotion: ', p_emo)
    print('Predicted phrase AV: ', label_phrase[np.argmax(prob_av[0])])
    print('Lead time video pre-processing: {} s'.format(np.round(end_pre_video, 2)))
    print('Lead time VER: {} s'.format(np.round(end_emo, 2)))
    print('Lead time VSR: {} s'.format(np.round(end_video, 2)))
    print('Lead time ASR: {} s'.format(np.round(end_audio, 2)))
    print('Lead time AVSR: {} s'.format(np.round(end_time, 2)))
    print('Lead time AVSR: {} s'.format(np.round(end_emo+end_video+end_audio+end_time+end_pre_video, 2)))
    print('Lead time ALL: {} s'.format(np.round(all_time_end, 2)))
    print()

def pred_all_video():
    path_all_videos = os.listdir(args.path_video)
    for id, cr_path in tqdm(enumerate(path_all_videos)):
        print('{}/{}'.format(id+1, len(path_all_videos)))
        pred_one_video(os.path.join(args.path_video,cr_path))
        
        
if __name__ == "__main__":
    pred_all_video()