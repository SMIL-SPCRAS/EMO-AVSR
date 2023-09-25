import librosa
import numpy as np

def get_log_mel(path):

    step_size = 128
    n_mels = 128
    len_spec = 251
    sec=2
    step=1
    sr=16000

    pl = [2048, step_size, None, True, 'reflect',
            2.0, n_mels, 'slaney', True, None]

    curr_features = []
    audio, sr = librosa.load(path, sr=sr)
    for id_, id_cur in enumerate(range(0, audio.shape[0]+1, sr*step)):
        win = sr*step*sec
        need_id = id_cur+win
        audio_curr = audio[id_cur:need_id]
        if len(audio_curr) > pl[0]:
            m_s = librosa.feature.melspectrogram(
                                y = audio_curr,
                                sr = sr,
                                n_fft = pl[0],
                                hop_length = pl[1],
                                win_length = pl[2],
                                center = pl[3],
                                pad_mode = pl[4],
                                power = pl[5],
                                n_mels = pl[6],
                                norm = pl[7],
                                htk = pl[8],
                                fmax = pl[9]
                            )

            db_m_s = librosa.power_to_db(m_s, top_db = 80)

            if db_m_s.shape[1] < len_spec:
                db_m_s = np.pad(db_m_s, ((0, 0), (0, len_spec-db_m_s.shape[1])), 'mean')

            db_m_s = db_m_s/255
            db_m_s = np.expand_dims(db_m_s, axis=-1)

            curr_features.append(db_m_s)
    return curr_features