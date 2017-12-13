import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_joint_spectrum(x, G, file, Nfft=None):
    G.compute_laplacian("normalized")
    G.compute_fourier_basis()
    xlambda = G.gft(x)
    T = x.shape[1]
    N = x.shape[0]
    if Nfft is not None:
        xflambda = np.fft.fft(xlambda, Nfft)
    else:
        xflambda = np.fft.fft(xlambda)
    xflambda = np.fft.fftshift(xflambda)

    fig = plt.figure()
    plt.imshow(20*np.log10(np.abs(xflambda)), origin="lower", extent=[-T//2, T//2, 0, N-1])
    plt.colorbar()
    plt.title("JFT(x)")
    plt.xlabel("f")
    plt.ylabel("lambda")
    fig.savefig(file + ".png")


def plot_temporal_matrix(x, file):
    fig = plt.figure()
    plt.imshow(x, origin="lower")
    plt.colorbar()
    plt.title("Signal x")
    plt.xlabel("t")
    plt.ylabel("v")
    fig.savefig(file + ".png")


if __name__ == '__main__':
    from pygsp import graphs
    from synthetic_data.data_generation import generate_spectral_samples_hard
    N = 100
    T = 128
    f_h = 50
    f_l = 15
    lambda_h = 80
    lambda_l = 15
    G = graphs.Community(N)
    x, _ = generate_spectral_samples_hard(1, T, G, f_h, lambda_h, f_l, lambda_l, sigma=2, sigma_n=1)
    # x = hp_hp_sample(T, G, f_h, lambda_h)
    plot_joint_spectrum(x[0, :, :, 0], G, "hp_hp")
    plot_temporal_matrix(x[0, :, :, 0], "hp_hp_t")
    # x = lp_lp_sample(T, G, f_l, lambda_l)
    plot_joint_spectrum(x[1, :, :, 0], G, "lp_lp")
    plot_temporal_matrix(x[1, :, :, 0], "lp_lp_t")
    # x = hp_lp_sample(T, G, f_h, lambda_l)
    plot_joint_spectrum(x[2, :, :, 0], G, "lp_hp")
    plot_temporal_matrix(x[2, :, :, 0], "lp_hp_t")
    # x = lp_hp_sample(T, G, f_l, lambda_h)
    plot_joint_spectrum(x[3, :, :, 0], G, "hp_lp")
    plot_temporal_matrix(x[3, :, :, 0], "hp_lp_t")
    # Combinations
    plot_joint_spectrum(x[4, :, :, 0], G, "lp_hp_hp_lp")
    plot_temporal_matrix(x[4, :, :, 0], "lp_hp_hp_lp_t")
    # x = lp_hp_sample(T, G, f_l, lambda_h)
    plot_joint_spectrum(x[5, :, :, 0], G, "hp_hp_lp_lp")
    plot_temporal_matrix(x[5, :, :, 0], "hp_hp_lp_lp_t")
