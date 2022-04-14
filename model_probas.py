import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, confusion_matrix

exp_dir = os.path.join('logs', 'mental', 'mental-bert-base-uncased_2')
probas_path = os.path.join(exp_dir, 'dev_probas.npy')

pred_provas, gt_labels = np.load(probas_path, allow_pickle=True)

neg_probas = pred_provas[gt_labels == 0]
pos_probas = pred_provas[gt_labels == 1]

plt.figure(figsize=(15, 10))
plt.hist(neg_probas, bins=40, color='red', alpha=0.5)
plt.hist(pos_probas, bins=40, color='blue', alpha=0.5)
plt.legend(['negatives', "positives"])
plt.show()


def threshold_predict(threshold):
    result_labels = np.zeros_like(gt_labels)
    pos_indices = np.where(pred_provas > threshold)[0]
    result_labels[pos_indices] = 1
    return result_labels

def get_best_confusion_matrix(best_threshold):
    result_labels = threshold_predict(best_threshold)
    conf_mat = confusion_matrix(gt_labels, result_labels)
    return conf_mat

def evaluate_threshold_model(threshold):
    result_labels = threshold_predict(threshold)

    f1_score_0 = f1_score(gt_labels, result_labels, average='binary', pos_label=0)
    f1_score_1 = f1_score(gt_labels, result_labels, average='binary', pos_label=1)
    # print(f"New result ended with f1_score = {f1_score_1} and considering 0 as the positive class = {f1_score_0}")
    return f1_score_1, f1_score_0

thresholds = np.linspace(0, 1, 100)

f1_list_0 = []
f1_list_1 = []
for threshold in thresholds:
    f1_score_1, f1_score_0 = evaluate_threshold_model(threshold)
    f1_list_0.append(f1_score_0)
    f1_list_1.append(f1_score_1)

plt.figure(figsize=(15, 10))
plt.plot(thresholds, f1_list_0)
plt.plot(thresholds, f1_list_1)
plt.legend(['f1_score_0', "f1_score_1"])
plt.show()

best_1_index = np.argmax(f1_list_1)
best_1_threshold = thresholds[best_1_index]

print(best_1_threshold, f1_list_1[best_1_index], f1_list_0[best_1_index])
print(get_best_confusion_matrix(best_1_threshold))





