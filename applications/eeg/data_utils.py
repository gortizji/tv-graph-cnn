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


def create_spatial_eeg_graph(montage, q=0.05, k=0.1):
    montage = mne.channels.read_montage(kind="standard_1005", ch_names=montage, unit="m")
    d = spdist.pdist(montage.pos)
    W = np.exp(- (d ** 2) / (2 * q ** 2))
    W[d > k] = 0
    W = spdist.squareform(W)
    G = graphs.Graph(W, lap_type="normalized", coords=montage.get_pos2d())
    print("Created EEG-graph with q=%.2f and k=%.2f" % (q, k))
    print("- Nodes:", G.N)
    print("- Edges:", G.Ne)
    return G


def create_data_eeg_graph(montage, X, threshold=0.7):
    montage = mne.channels.read_montage(kind="standard_1005", ch_names=montage, unit="m")
    C = np.matmul(X, X.T) / X.shape[1]
    C = C - np.eye(C.shape[0])
    C[C < threshold] = 0
    G = graphs.Graph(C, lap_type="normalized", coords=montage.get_pos2d())
    print("- Nodes:", G.N)
    print("- Edges:", G.Ne)
    return G


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
    G = create_spatial_eeg_graph(MONTAGE, q, k)
    plotting.plot_graph(G, "matplotlib", save_as="graph")
    fig = plt.figure()
    plt.imshow(G.W.todense())
    plt.colorbar()
    fig.savefig("W.jpg")
    # plot_spectrum(3, 2, q, k)
