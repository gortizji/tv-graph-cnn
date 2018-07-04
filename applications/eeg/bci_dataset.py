import os

import numpy as np
import scipy
import scipy.signal as spsig
from scipy import io as spio

FILEDIR = os.path.dirname(os.path.realpath(__file__))
BCI_IV_DIR = os.path.join(FILEDIR, "datasets/BCI_IV_2a")
SAMPLING_FREQUENCY = 250
TRIALS_PER_RUN = 48
TRIAL_SUBDIVISIONS = 4
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
           "POZ"]
           #"FP1", "NZ", "FP2"]   # Ordered as in BCI IV 2a dataset


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


def load_run_from_subject(subject_data, run_number, start_runs=3):
    assert run_number < RUNS_PER_SUBJECT
    run = subject_data[run_number+start_runs]  # First three runs are meaningless
    return run


def load_run_data(run):
    return (run.X).T  # (nodes, samples)


def load_run_labels(run):
    return run.y - 1


def load_run_markers(run):
    return run.trial


def get_trial(run, sample_number):
    assert sample_number < (TRIALS_PER_RUN * TRIAL_SUBDIVISIONS)
    markers = load_run_markers(run)
    trial_number = sample_number // TRIAL_SUBDIVISIONS
    start = markers[trial_number]
    y = load_run_labels(run)[trial_number]
    X = load_run_data(run)
    sub_offset = 2 + (4 / TRIAL_SUBDIVISIONS) * (sample_number % TRIAL_SUBDIVISIONS)
    X_crop = X[:, (start + 2 * SAMPLING_FREQUENCY):(start + sub_offset * SAMPLING_FREQUENCY)]  # Trial data between 2 and 6 secs
    X_crop = X_crop[:-3, :]
    b, a = spsig.butter(3, Wn=4 / SAMPLING_FREQUENCY, analog=False, output="ba", btype="high")
    X_crop = spsig.filtfilt(b, a, X_crop, axis=1)
    #X_crop = X_crop - np.tile(np.expand_dims(np.mean(X_crop, axis=1), axis=-1), (1, X_crop.shape[1]))
    X_crop = X_crop / np.tile(np.expand_dims(np.std(X_crop, axis=1), axis=-1), (1, X_crop.shape[1]))
    return X_crop, y


def get_subject_dataset(subject_id, training=True):
    subject_data = load_subject(subject_id, training)
    data = []
    labels = []
    for run_number in range(RUNS_PER_SUBJECT):
        run = load_run_from_subject(subject_data, run_number, start_runs=1 if (subject_id == 3 and training) else 3)
        for trial in range(TRIALS_PER_RUN * TRIAL_SUBDIVISIONS):
            X, y = get_trial(run, trial)
            data.append(X)
            labels.append(y)
    data = np.array(data)
    data = np.expand_dims(data, axis=-1)
    labels = np.array(labels)
    return data, labels


def get_full_dataset(training=True):
    data = []
    labels = []
    for id in range(NUM_SUBJECTS):
        X_s, y_s = get_subject_dataset(id, training)
        data.append(X_s)
        labels.append(y_s)
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    return data, labels