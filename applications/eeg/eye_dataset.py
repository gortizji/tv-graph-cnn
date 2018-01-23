import os
import numpy as np
from scipy.io import arff
import scipy.signal as spsig

from sklearn.model_selection import KFold

from tv_graph_cnn.minibatch_sources import MinibatchSource
from applications.eeg.data_utils import plot_montage, create_spatial_eeg_graph

FILEDIR = os.path.dirname(os.path.realpath(__file__))
EYE_FILE = os.path.join(FILEDIR, "datasets/EEG_eye.arff")

MONTAGE = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

SAMPLING_FREQUENCY = 128


class EyeMinibatchSource(MinibatchSource):

    def __init__(self, data, labels, sample_idx, margin, repeat=False):
        self.margin = margin
        self.sample_idx = sample_idx  # Samples from data belonging to data partition
        super().__init__(data, labels, repeat)

    @property
    def dataset_length(self):
        return self.sample_idx.shape[0]

    def _data_chunk(self, start=None, end=None):
        if start is None:
            times = self.sample_idx[self.indices[:end]]
        elif end is None:
            times = self.sample_idx[self.indices[start:]]
        else:
            times = self.sample_idx[self.indices[start:end]]

        X = []
        for t in times:
            x_t = get_snapshot(self.data, t, self.margin)
            X.append(x_t)
        X = np.array(X)
        return X

    def _labels_chunk(self, start=None, end=None):
        if start is None:
            return self.labels[self.sample_idx[self.indices[:end]]]
        elif end is None:
            return self.labels[self.sample_idx[self.indices[start:]]]
        else:
            return self.labels[self.sample_idx[self.indices[start:end]]]


def load_data():
    global MONTAGE
    data, meta = arff.loadarff(EYE_FILE)
    # MONTAGE = meta.names()[:-1]  # Last attribute is label
    labels = [int(sample[-1]) for sample in data]
    labels = np.array(labels)
    X = [[node for node in sample] for sample in data]
    X = np.array(X, dtype=np.float32)
    X = X[:, :-1].T
    X = np.delete(X, (898, 10386, 11509, 13179), axis=1)
    labels = np.delete(labels, (898, 10386, 11509, 13179), axis=0)
    b, a = spsig.butter(4, Wn=0.5 / SAMPLING_FREQUENCY, analog=False, output="ba", btype="high")
    X_filtered = spsig.filtfilt(b, a, X, axis=1)
    #X_filtered = X - np.tile(np.expand_dims(np.mean(X, axis=1), axis=-1), (1, X.shape[1]))
    X = X_filtered / np.tile(np.expand_dims(np.std(X_filtered, axis=1), axis=-1), (1, X.shape[1]))
    return X, labels


def train_validation_test_split(dataset_length, margin, test_size, validation_size=0, N_chunks=50):
    chunk_idx = np.arange(0, N_chunks)
    samples_per_chunk = dataset_length // N_chunks

    N_test = np.floor(test_size * N_chunks).astype(int)
    N_validation = np.floor(validation_size * N_chunks).astype(int)

    test_idx = list(np.random.choice(chunk_idx, size=N_test))
    if N_validation > 0:
        validation_idx = np.random.choice([idx for idx in chunk_idx if idx not in test_idx], size=N_validation)
    else:
        validation_idx = []

    train_idx = [idx for idx in chunk_idx if idx not in (test_idx + validation_idx)]

    test_idx = np.array(test_idx)
    train_idx = np.array(train_idx)
    validation_idx = np.array(validation_idx)

    def _get_samples_from_chunks(times, chunk_list):
        samples = []
        for idx in chunk_list:
            samples_idx = times[idx * samples_per_chunk: (idx + 1) * samples_per_chunk]
            samples.append(samples_idx)
        samples = np.concatenate(samples, axis=0)
        return samples

    def _clean_sample_indices(samples, margin):
        samples_cleaned = [sample for sample in samples if
                           all(t in samples for t in (sample - margin, sample + margin - 1))]
        return np.array(samples_cleaned)

    times = np.arange(dataset_length)

    train_times = _get_samples_from_chunks(times, train_idx)
    print(train_times.shape[0])
    train_times = _clean_sample_indices(train_times, margin)
    test_times = _get_samples_from_chunks(times, test_idx)
    test_times = _clean_sample_indices(test_times, margin)

    if N_validation > 0:
        validation_times = _get_samples_from_chunks(times, validation_idx)
        validation_times = _clean_sample_indices(validation_times, margin)
        return train_times, validation_times, test_times
    else:
        return train_times, test_times


# def train_validation_test_split(X, y, test_size, validation_size=0):
#     dataset_length = len(y)
#     if test_size < 1:
#         start_val = int(dataset_length * (1 - validation_size - test_size))
#         start_test = int(dataset_length * (1 - test_size))
#     else:
#         start_val = dataset_length - validation_size - test_size
#         start_test = dataset_length - test_size
#
#     X_train = X[:, :start_val]
#     y_train = y[:start_val]
#
#     X_test = X[:, start_test:]
#     y_test = y[start_test:]
#
#     if validation_size > 0:
#         X_val = X[:, start_val:start_test]
#         y_val = y[start_val:start_test]
#
#         return X_train, y_train, X_val, y_val, X_test, y_test
#     else:
#         return X_train, y_train, X_test, y_test


def get_snapshot(X, t, margin):
    return X[:, t - margin:t + margin]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from pygsp import plotting
    import scipy.io as sio

    from graph_utils.visualization import plot_joint_spectrum, plot_temporal_matrix
    from applications.eeg.data_utils import create_data_eeg_graph
    import mne

    X, labels = load_data()

    info = mne.create_info(
        ch_names=MONTAGE,
        ch_types=["eeg"] * len(MONTAGE),
        sfreq=SAMPLING_FREQUENCY
    )

    raw = mne.io.RawArray(X, info)

    print(X.shape)
    print(X.nbytes / 1024)
    plot_montage(MONTAGE)

    fig = plt.figure()
    plt.plot(labels)
    fig.savefig("labels.png")

    C = np.matmul(X, X.T) / X.shape[1]

    X_1 = X[:, labels > 0]
    X_0 = X[:, labels < 1]
    C_0 = np.matmul(X_0, X_0.T) / X_0.shape[1]
    C_1 = np.matmul(X_1, X_1.T) / X_1.shape[1]

    l, v = np.linalg.eig(C)
    l_0, v_0 = np.linalg.eig(C_0)
    l_1, v_1 = np.linalg.eig(C_1)

    P_0 = np.matmul(np.matmul(v_0.T, C_0), v_0)
    P_1 = np.matmul(np.matmul(v_1.T, C_1), v_1)
    P = np.matmul(np.matmul(v.T, C), v)
    print("P_0: %1.3f, P_1: %1.3f" % (np.matrix.trace(P_0), np.matrix.trace(P_1)))

    G_rbf = create_spatial_eeg_graph(MONTAGE, q=0.1, k=0.1)
    G_rbf.compute_laplacian("normalized")
    G_rbf.compute_fourier_basis()
    U_rbf = G_rbf.U
    plotting.plot_graph(G_rbf, "matplotlib", save_as="grap")

    fig = plt.figure()
    plt.imshow(G_rbf.W.todense())
    plt.colorbar()
    fig.savefig("W.png")


    P_0_rbf = np.matmul(np.matmul(U_rbf.T, C_0), U_rbf)
    P_1_rbf = np.matmul(np.matmul(U_rbf.T, C_1), U_rbf)
    P_rbf = np.matmul(np.matmul(U_rbf.T, C), U_rbf)

    fig = plt.figure()
    plt.imshow(P_0_rbf)
    plt.colorbar()
    fig.savefig("P_0_rbf.png")

    fig = plt.figure()
    plt.imshow(P_1_rbf)
    plt.colorbar()
    fig.savefig("P_1_rbf.png")

    fig = plt.figure()
    plt.imshow(P_rbf)
    plt.colorbar()
    fig.savefig("P_rbf.png")

    fig = plt.figure()
    plt.plot(l)
    fig.savefig("eig.png")

    C = C - np.eye(C.shape[0])
    C[C < 0.7] = 0
    fig = plt.figure()
    plt.imshow(C)
    plt.colorbar()
    fig.savefig("Correlation.png")

    G = create_data_eeg_graph(MONTAGE, X)
    plotting.plot_graph(G, "matplotlib", save_as="grap_corr")

    t = 3850
    X_s = get_snapshot(X, t, 64)
    plot_joint_spectrum(X_s, G_rbf, "jtv_corr_1")

    montage = mne.channels.read_montage(kind="standard_1005", ch_names=MONTAGE, unit="m")

    sio.savemat("eye_EEG.mat", {"X": X,
                                "X1": X_1,
                                "X0": X_0,
                                "L_rbf": G.L,
                                "Rx": C,
                                "Rx1": C_1,
                                "Rx0": C_0,
                                "montage": MONTAGE,
                                "coords2d": montage.get_pos2d(),
                                "coords3d": montage.pos,
                                "labels": labels
                                })
