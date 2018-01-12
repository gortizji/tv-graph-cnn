import scipy.spatial.distance as spdist
import scipy.signal as spsig
import numpy as np
import matplotlib

from applications.eeg.bci_dataset import SAMPLING_FREQUENCY, MONTAGE, load_subject, load_run_from_subject, get_trial, \
    get_subject_dataset

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
from graph_utils.visualization import plot_joint_spectrum, plot_temporal_matrix
from pygsp import graphs


def plot_montage(montage):
    montage = mne.channels.read_montage(kind="standard_1005", ch_names=montage, unit="m")
    fig = montage.plot(scale_factor=10)
    fig.savefig("montage.jpg")


def create_eeg_graph(montage, q=0.05, k=0.1):
    montage = mne.channels.read_montage(kind="standard_1005", ch_names=montage, unit="m")
    d = spdist.pdist(montage.pos)
    W = np.exp(- (d**2) / (2*q**2))
    W[d>k] = 0
    W = spdist.squareform(W)
    G = graphs.Graph(W, lap_type="normalized", coords=montage.get_pos2d())
    print("Created EEG-graph with q=%.2f and k=%.2f" % (q, k))
    print("- Nodes:", G.N)
    print("- Edges:", G.Ne)
    return G


def plot_spectrum(class_id, subject_id, q=0.05, k=0.1):
    data, labels = get_subject_dataset(subject_id, training=True)
    data_class = data[labels == class_id]
    G = create_eeg_graph(q, k)
    idx = 0
    b, a = spsig.butter(3, 4 / SAMPLING_FREQUENCY, btype="highpass", output="ba")
    data_filtered = spsig.lfilter(b, a, data_class[idx, :, :, 0])
    plot_joint_spectrum(data_filtered, G, "jtv_" + str(class_id) + "_" + str(q) + "_" + str(k))
    plot_temporal_matrix(data_filtered, "matrix_" + str(class_id) + "_" + str(q) + "_" + str(k))


if __name__ == '__main__':
    from pygsp import plotting
    from applications.eeg.eye_dataset import MONTAGE
    s = load_subject(8, True)
    r = load_run_from_subject(s, 5)
    X, y = get_trial(r, 47)
    data, labels = get_subject_dataset(1, False)
    print(data.shape)
    q = 0.1
    k = 0.15
    plot_montage(MONTAGE)
    G = create_eeg_graph(MONTAGE, q, k)
    plotting.plot_graph(G, "matplotlib", save_as="graph")
    fig = plt.figure()
    plt.imshow(G.W.todense())
    plt.colorbar()
    fig.savefig("W.jpg")
    #plot_spectrum(3, 2, q, k)
