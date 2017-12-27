import numpy as np


class MinibatchSource:
    def __init__(self, data, labels, repeat=False):
        self.data = data
        self.labels = labels
        self.indices = np.random.permutation(self.dataset_length)
        self.idx = 0
        self.repeat = repeat
        self.end = False

    @property
    def dataset_length(self):
        return self.data.shape[0]

    def next_batch(self, batch_size):
        if self.end:
            if not self.repeat:
                return None, self.end
            else:
                self.end = False

        end_idx = self.idx + batch_size

        if end_idx < self.dataset_length:
            batch = [self.data[self.indices[self.idx:end_idx]], self.labels[self.indices[self.idx:end_idx]]]
        else:
            self.end = True
            batch = [self.data[self.indices[self.idx:]], self.labels[self.indices[self.idx:]]]
            if self.repeat:
                end_idx = end_idx % self.dataset_length
                self.indices = np.random.permutation(self.dataset_length)
                batch[0] = np.concatenate((batch[0], self.data[self.indices[:end_idx]]), axis=0)
                batch[1] = np.concatenate((batch[1], self.labels[self.indices[:end_idx]]), axis=0)
        self.idx = end_idx

        return batch, self.end

    def restart(self):
        self.end = False
        self.idx = 0