import numpy as np
import h5py
import os

from pygsp import graphs

from sklearn.model_selection import train_test_split

from tv_graph_cnn.minibatch_sources import MinibatchSource
from graph_utils.coarsening import perm_data


FILEDIR = os.path.dirname(os.path.realpath(__file__))
DATASET_FILE = os.path.join(FILEDIR, "datasets/SEIRS.mat")


class EpidemyHDF5MinibatchSource(MinibatchSource):

    def __init__(self, hdf5_file, samples, repeat=False, perm=None, samples_memory_size=100, runs_same_memory=5):

        self.hdf5_file = hdf5_file
        self.perm = perm

        self.samples = samples
        if samples_memory_size < self.all_dataset_length:
            self.samples_memory_size = samples_memory_size
            self.runs_same_memory = runs_same_memory
        else:
            self.samples_memory_size = self.all_dataset_length
            self.runs_same_memory = np.inf

        self.run_counts = 0
        self.max_runs = np.ceil(self.all_dataset_length / self.samples_memory_size) if not repeat else np.inf

        if repeat:
            self.current_samples = np.sort(np.random.choice(self.samples, self.samples_memory_size, replace=False))
        else:
            self.current_samples = np.sort(self.samples[:self.samples_memory_size])

        with h5py.File(self.hdf5_file) as f:
            self.hdf5_length = _hdf5_length(f)
            X, cp, Tim, _ = _unpack_variables(f)

            X_current, y_current = self._load_current_data(X, cp, Tim)

        super(EpidemyHDF5MinibatchSource, self).__init__(X_current, y_current, repeat=repeat)
        print("Created MinibatchSource with %d samples out of %d!" % (self.all_dataset_length, self.hdf5_length))

    def _load_current_data(self, X, cp, Tim):
        if self.perm is not None:
            X_current = perm_data(X[self.current_samples, :, :], self.perm)
        else:
            X_current = X[self.current_samples, :, :]

        y = _combine_params(cp, Tim)
        y_current = y[self.current_samples]

        #mean_X = np.mean(X_current, axis=(1, 2))
        #X_current = X_current - mean_X[:, np.newaxis, np.newaxis]
        return X_current, y_current

    @property
    def all_dataset_length(self):
        return len(self.samples)

    @property
    def time_length(self):
        return self.data.shape[2]

    def next_batch(self, batch_size):
        batch, _ = super(EpidemyHDF5MinibatchSource, self).next_batch(batch_size)
        if self.end:
            self.run_counts += 1

            if self.repeat:
                if self.run_counts >= self.runs_same_memory:
                    self._update_current_samples()
                    self.run_counts = 0

            else:
                if self.run_counts < self.max_runs:
                    self._update_current_samples()
                    self.end = False
                    self.idx = 0
                    if batch_size - len(batch[0]) > 0:
                        batch_missing = super(EpidemyHDF5MinibatchSource, self).next_batch(batch_size - len(batch[0]))
                        batch[0] = np.concatenate([batch[0], batch_missing[0]], axis=0)
                        batch[1] = np.concatenate([batch[1], batch_missing[1]], axis=0)

        return batch, self.end

    def _update_current_samples(self):
        if self.repeat:
            self.current_samples = np.sort(np.random.choice(self.samples, self.samples_memory_size, replace=False))
        else:
            if (self.run_counts + 1) * self.samples_memory_size >= self.all_dataset_length:
                self.current_samples = np.sort(self.samples[self.run_counts * self.samples_memory_size:])
            else:
                self.current_samples = np.sort(self.samples[self.run_counts * self.samples_memory_size:(self.run_counts + 1) * self.samples_memory_size])

        with h5py.File(self.hdf5_file) as f:
            X, cp, Tim, A = _unpack_variables(f)
            self.data, self.labels = self._load_current_data(X, cp, Tim)


def _combine_params(cp, Tim):
    return np.concatenate([np.array(cp), np.array(Tim)], axis=1)


def _unpack_variables(f):
    cp = f["cp"]
    Tim = f["Tim"]
    A = f["A"]
    X = f["X"]
    mean_X = np.mean(X, axis=(1, 2))
    cp = np.log10(cp)
    mean_cp = np.mean(cp)
    std_cp = np.std(cp)
    mean_tim = np.mean(Tim)
    std_tim = np.std(Tim)
    X = X - mean_X[:, np.newaxis, np.newaxis]
    cp = (cp - mean_cp) / std_cp
    Tim = (Tim - mean_tim) / std_tim
    return X, cp, Tim, A


def _hdf5_length(f):
    _, cp, _, _ = _unpack_variables(f)
    return len(cp)


def create_graph(hdf5_file):
    with h5py.File(hdf5_file, "r") as f:
        _, _, _, A = _unpack_variables(f)
        G = graphs.Graph(A, lap_type="normalized")
    return G


def create_train_test_mb_sources(hdf5_file, test_size, perm=None, samples_memory_size=10000, runs_same_memory=2):
    with h5py.File(hdf5_file, "r") as f:
        n_samples = _hdf5_length(f)

    idx_train, idx_test = train_test_split(np.arange(n_samples), test_size=test_size)
    print("Creating train MinibatchSource...")
    train_mb_source = EpidemyHDF5MinibatchSource(hdf5_file, idx_train,
                                                 repeat=True,
                                                 perm=perm,
                                                 samples_memory_size=samples_memory_size,
                                                 runs_same_memory=runs_same_memory)
    print("Creating test MinibatchSource...")
    test_mb_source = EpidemyHDF5MinibatchSource(hdf5_file, idx_test,
                                                repeat=False,
                                                perm=perm,
                                                samples_memory_size=samples_memory_size)
    print("MinibatchSources created")

    return train_mb_source, test_mb_source


if __name__ == '__main__':
    a, b = create_train_test_mb_sources(DATASET_FILE, 0.2)

