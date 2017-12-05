import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.signal import square


def delta(Nv, c):
    d = np.zeros([Nv, 1])
    d[c] = 1
    return d


def diffusion_seq(W, x, T):
    N = W.shape[0]
    X = np.zeros((N, T))
    for t in range(T):
        if t == 0:
            X[:, 0] = x.T / np.max(x)
            continue
        X[:, t] = ((W ** t) * X[:,t-1]).T
        X[:, t] /= np.max(X)
    return X


def generate_diffusion_samples(N, W, T=10, sigma=0.05):
    samples = []
    labels = []
    Nv = W.shape[0]
    for _ in range(N):
        k = np.random.randint(Nv)
        Wk = W ** k
        c = np.random.randint(Nv)
        x = Wk.dot(delta(Nv, c))
        X = diffusion_seq(W, x, T) + sigma * np.random.randn(Nv, T)
        samples.append(X)
        labels.append(c)
    return np.array(samples), np.array(labels)


def wave_seq(Wphi, b, Kstart, T, f):
    Nv = Wphi.shape[0]
    X = np.zeros((Nv, Kstart+T))
    phi = 2 * np.pi * np.random.rand()
    duty = np.random.rand()
    X[:,0] = (b * 0.5*(1-square(phi, duty=duty))).T
    for t in range(1, Kstart+T):
        X[:,t] = Wphi.dot(X[:,t-1]).T + (b *  0.5 * (1-square(f*t+phi, duty=duty))).T
    return X[:, Kstart:Kstart+T]


def generate_wave_samples(N, W, T=10, Kmin=10, Kmax=100, sigma=0.05):
    samples = []
    labels = []
    Nv = W.shape[0]
    _, U = eigsh(W)
    Wphi = U.dot(U.T)
    for _ in range(N):
        kstart = np.random.randint(Kmin, Kmax)
        c = np.random.randint(Nv)
        b = delta(Nv, c)
        f = np.pi*np.random.rand()
        X = wave_seq(Wphi, b, kstart, T, f) + sigma * np.random.randn(Nv, T)
        samples.append(X)
        labels.append(c)
    return np.array(samples), np.array(labels)

