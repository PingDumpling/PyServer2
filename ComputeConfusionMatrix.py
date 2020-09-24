import numpy as np


def comp_conf_mat(y_true, y_pred):
    set_label = set(y_true)
    cm = np.zeros((len(set_label), len(set_label)))
    for i in range(len(y_true)):
        cm[int(y_true[i])][int(y_pred[i])] += 1
    return cm


#def wp_acc(cm):


