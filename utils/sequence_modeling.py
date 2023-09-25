import tensorflow as tf


def get_sequence(all_path, all_feature, win = 10, step = 5):
    seq_path = []
    seq_feature = []
    for id_cur in range(0, len(all_path)+1, step):
        need_id = id_cur+win
        curr_FE = all_feature[id_cur:need_id]
        if len(curr_FE) < win and len(curr_FE) != 0:
            curr_FE = tf.concat([curr_FE, [curr_FE[-1]]*(win - len(curr_FE))], 0)
        if len(curr_FE) != 0:
            seq_feature.append(curr_FE)
    return seq_feature