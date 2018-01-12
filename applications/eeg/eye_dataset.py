import os
import numpy as np
from scipy.io import arff

from tv_graph_cnn.minibatch_sources import MinibatchSource
from applications.eeg.data_utils import plot_montage

FILEDIR = os.path.dirname(os.path.realpath(__file__))
EYE_FILE = os.path.join(FILEDIR, "datasets/EEG_eye.arff")

MONTAGE = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']


class EyeMinibatchSource(MinibatchSource):

    def __init__(self, data, labels, margin, repeat=False):
        self.margin = margin
        super().__init__(data, labels, repeat)

    @property
    def dataset_length(self):
        return self.data.shape[1] - (2 * self.margin)  # Snapshots for [t_start=margin, t_end=end-margin]

    def _data_chunk(self, start=None, end=None):
        if start is None:
            times = self.indices[:end]
        elif end is None:
            times = self.indices[start:]
        else:
            times = self.indices[start:end]

        X = []
        for t in times:
            x_t = get_snapshot(self.data, t + self.margin, self.margin)  # First snapshot t=margin
            X.append(x_t)
        X = np.array(X)
        return X

    def _labels_chunk(self, start=None, end=None):
        if start is None:
            return self.labels[self.indices[:end] + self.margin]  # First snapshot t=margin
        elif end is None:
            return self.labels[self.indices[start:] + self.margin]  # First snapshot t=margin
        else:
            return self.labels[self.indices[start:end] + self.margin]  # First snapshot t=margin


def load_data():
    global MONTAGE
    data, meta = arff.loadarff(EYE_FILE)
    # MONTAGE = meta.names()[:-1]  # Last attribute is label
    labels = [int(sample[-1]) for sample in data]
    labels = np.array(labels)
    X = [[node for node in sample] for sample in data]
    X = np.array(X, dtype=np.float32)
    X = X[:, :-1].T

    X = X - np.tile(np.expand_dims(np.mean(X, axis=1), axis=-1), (1, X.shape[1]))
    X = X / np.tile(np.expand_dims(np.std(X, axis=1), axis=-1), (1, X.shape[1]))
    return X, labels


def train_validation_test_split(X, y, test_size, validation_size=0):
    dataset_length = len(y)
    if test_size < 1:
        start_val = int(dataset_length * (1 - validation_size - test_size))
        start_test = int(dataset_length * (1 - test_size))
    else:
        start_val = dataset_length - validation_size - test_size
        start_test = dataset_length - test_size

    X_train = X[:, :start_val]
    y_train = y[:start_val]

    X_test = X[:, start_test:]
    y_test = y[start_test:]

    if validation_size > 0:
        X_val = X[:, start_val:start_test]
        y_val = y[start_val:start_test]

        return X_train, y_train, X_val, y_val, X_test, y_test
    else:
        return X_train, y_train, X_test, y_test


def get_snapshot(X, t, margin):
    return X[:, t-margin:t+margin]


if __name__ == '__main__':
    X, labels = load_data()
    print(X.shape)
    print(X.nbytes/1024)
    plot_montage(MONTAGE)

