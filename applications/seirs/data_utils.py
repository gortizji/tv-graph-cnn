import numpy as np
import h5py
import os

from sklearn.model_selection import train_test_split

from tv_graph_cnn.minibatch_sources import MinibatchSource


FILEDIR = os.path.dirname(os.path.realpath(__file__))
DATASET_FILE = os.path.join(FILEDIR, "datasets/SEIRS.mat")


class EpidemyHDF5MinibatchSource(MinibatchSource):

    def __init__(self, hdf5_file, samples, repeat=False, samples_memory_size=100, runs_same_memory=5):

        self.hdf5_file = hdf5_file

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

            X_current = X[self.current_samples, :, :]
            y = _combine_params(cp, Tim)

        super(EpidemyHDF5MinibatchSource, self).__init__(X_current, y[self.current_samples], repeat=repeat)
        print("Created!")

    @property
    def all_dataset_length(self):
        return len(self.samples)

    def next_batch(self, batch_size):
        batch, _ = super(EpidemyHDF5MinibatchSource, self).next_batch(batch_size)
        if self.end:
            self.run_counts += 1

            if self.repeat:
                if self.run_counts >= self.runs_same_memory:
                    self.update_current_samples()
                    self.run_counts = 0

            else:
                self.update_current_samples()
                if self.run_counts < self.max_runs:
                    self.update_current_samples()
                    self.end = False
                    self.idx = 0
                    if batch_size - len(batch[0]) > 0:
                        batch_missing = super(EpidemyHDF5MinibatchSource, self).next_batch(batch_size - len(batch[0]))
                        batch[0] = np.concatenate([batch[0], batch_missing[0]], axis=0)
                        batch[1] = np.concatenate([batch[1], batch_missing[1]], axis=0)

        return batch, self.end

    def update_current_samples(self):
        if self.repeat:
            self.current_samples = np.sort(np.random.choice(self.samples, self.samples_memory_size, replace=False))
        else:
            if (self.run_counts + 1) * self.samples_memory_size >= self.all_dataset_length:
                self.current_samples = np.sort(self.samples[self.run_counts * self.samples_memory_size:])
            else:
                self.current_samples = np.sort(self.samples[self.run_counts * self.samples_memory_size:(self.run_counts + 1) * self.samples_memory_size])

        with h5py.File(self.hdf5_file) as f:
            X, cp, Tim, A = _unpack_variables(f)
            self.data = np.array(X[self.current_samples, :, :])
            self.labels = _combine_params(cp, Tim)[self.current_samples]


def _combine_params(cp, Tim):
    return np.concatenate([np.array(cp), np.array(Tim)], axis=1)


def _unpack_variables(f):
    cp = f["cp"]
    Tim = f["Tim"]
    A = f["A"]
    X = f["X"]
    return X, cp, Tim, A


def _hdf5_length(f):
    _, cp, _, _ = _unpack_variables(f)
    return len(cp)


def create_train_test_mb_sources(hdf5_file, test_size, samples_memory_size=5000, runs_same_memory=20):
    with h5py.File(hdf5_file, "r") as f:
        n_samples = _hdf5_length(f)

    idx_train, idx_test = train_test_split(np.arange(n_samples), test_size=test_size)
    print("Creating train MinibatchSource...")
    train_mb_source = EpidemyHDF5MinibatchSource(hdf5_file, idx_train, repeat=True, samples_memory_size=samples_memory_size, runs_same_memory=runs_same_memory)
    print("Creating test MinibatchSource...")
    test_mb_source = EpidemyHDF5MinibatchSource(hdf5_file, idx_test, repeat=False, samples_memory_size=samples_memory_size)
    print("MinibatchSources created")

    return train_mb_source, test_mb_source


if __name__ == '__main__':
    a, b = create_train_test_mb_sources(DATASET_FILE, 0.1)

