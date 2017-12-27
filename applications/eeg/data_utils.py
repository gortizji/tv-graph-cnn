import scipy
import scipy.io as spio
import scipy.spatial.distance as spdist
import numpy as np
import os
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
from pygsp import graphs


FILEDIR = os.path.dirname(os.path.realpath(__file__))
BCI_IV_DIR = os.path.join(FILEDIR, "datasets/BCI_IV_2a")
SAMPLING_FREQUENCY = 250
TRIALS_PER_RUN = 48
RUNS_PER_SUBJECT = 6
NUM_SUBJECTS = 9
TRIAL_LENGTH = 4
NUM_CLASSES = 4
NUM_TRIALS = RUNS_PER_SUBJECT * TRIALS_PER_RUN
SAMPLES_PER_TRIAL = SAMPLING_FREQUENCY * TRIAL_LENGTH
MONTAGE = ["FZ",
           "FC3", "FC1", "FCZ", "FC2", "FC4",
           "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
           "CP3", "CP1", "CPZ", "CP2", "CP4",
           "P1", "PZ", "P2",
           "POZ",
           "FP1", "NZ", "FP2"]   # Ordered as in BCI IV 2a dataset


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def load_subject(subject_id, training=True):
    assert subject_id < NUM_SUBJECTS
    subject_file = "A0" + str(subject_id + 1) + ("T" if training else "E") + ".mat"
    mat_variables = loadmat(os.path.join(BCI_IV_DIR, subject_file))
    subject_data = mat_variables["data"]
    return subject_data


def load_run_from_subject(subject_data, run_number):
    assert run_number < RUNS_PER_SUBJECT
    run = subject_data[run_number+3]  # First three runs are meaningless
    return run


def load_run_data(run):
    return (run.X).T  # (nodes, samples)


def load_run_labels(run):
    return run.y


def load_run_markers(run):
    return run.trial


def get_trial(run, trial_number):
    assert trial_number < TRIALS_PER_RUN
    markers = load_run_markers(run)
    start = markers[trial_number]
    y = load_run_labels(run)[trial_number]
    X = load_run_data(run)
    X_crop = X[:, (start + 2 * SAMPLING_FREQUENCY):(start + 6 * SAMPLING_FREQUENCY)]  # Trial data between 2 and 6 secs
    return X_crop, y


def get_subject_dataset(subject_id, training=True):
    subject_data = load_subject(subject_id, training)
    data = []
    labels = []
    for run_number in range(RUNS_PER_SUBJECT):
        run = load_run_from_subject(subject_data, run_number)
        for trial in range(TRIALS_PER_RUN):
            X, y = get_trial(run, trial)
            data.append(X)
            labels.append(y)
    data = np.array(data)
    data = np.expand_dims(data, axis=-1)
    labels = np.array(labels)
    return data, labels


def plot_montage():
    montage = mne.channels.read_montage(kind="standard_1005", ch_names=MONTAGE, unit="m")
    fig = montage.plot(scale_factor=10)
    fig.savefig("montage.jpg")


def create_eeg_graph(q=0.05, k=0.1):
    montage = mne.channels.read_montage(kind="standard_1005", ch_names=MONTAGE, unit="m")
    d = spdist.pdist(montage.pos)
    W = np.exp(- (d**2) / (2*q**2))
    W[d>k] = 0
    W = spdist.squareform(W)
    G = graphs.Graph(W, lap_type="normalized", coords=montage.get_pos2d())
    print("Created EEG-graph with q=%.2f and k=%.2f" % (q, k))
    print("- Nodes:", G.N)
    print("- Edges:", G.Ne)
    return G


if __name__ == '__main__':
    s = load_subject(8, True)
    r = load_run_from_subject(s, 5)
    X, y = get_trial(r, 47)
    data, labels = get_subject_dataset(1, False)
    print(data.shape)
    plot_montage()
    G = create_eeg_graph()
    fig = plt.figure()
    plt.imshow(G.W.todense())
    fig.savefig("W.jpg")
