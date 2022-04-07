import numpy as np
import matplotlib.pyplot as plt
import pdb

from __init__ import time_series_data_path

def plot_labels_distro(all_labels, n_bins=2):
    array = plt.hist(all_labels, bins=n_bins)
    for i in range(n_bins):
        plt.text(array[1][i], array[0][i], str(int(array[0][i])))
    plt.show()

def plot_time_series_seqs_by_label(train_text, train_labels, n_series=2):
    for label in range(2):
        plt.figure(figsize=(15, 10))
        plt.title(f"Time series for label = {label}")
        labeled_text = train_text[train_labels == label]
        for train_seq in labeled_text[:n_series]:
            plt.plot(np.arange(len(train_seq)), train_seq)
        plt.show()

def plot_time_series_seqs_comparing(train_text, train_labels):
    plt.figure(figsize=(15, 10))
    for label in range(2):
        train_seq = train_text[train_labels == label]
        random_index = np.random.randint(len(train_seq))
        plt.plot(np.arange(len(train_seq[random_index])), train_seq[random_index])
    plt.legend([f"Label {i} sample evolution" for i in range(2)])
    plt.show()

def main():
    train_text, dev_text, train_labels, dev_labels = np.load(time_series_data_path, allow_pickle=True)
    all_labels = np.concatenate([train_labels, dev_labels])
    plot_labels_distro(all_labels)
    plot_time_series_seqs_by_label(train_text, train_labels)
    plot_time_series_seqs_comparing(train_text, train_labels)


if __name__ == "__main__":
    main()