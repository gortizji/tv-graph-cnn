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
    plt.imshow(np.abs(xflambda)**2, origin="lower", extent=[-T//2, T//2, 0, N-1])
    plt.title("JFT(x)")
    plt.xlabel("f")
    plt.ylabel("lambda")
    fig.savefig(file + ".png")


if __name__ == '__main__':
    from pygsp import graphs
    from synthetic_data.data_generation import hp_hp_sample, hp_lp_sample, lp_hp_sample, lp_lp_sample
    N = 100
    T = 128
    f_h = 20
    f_l = 40
    lambda_h = 40
    lambda_l = 60
    G = graphs.ErdosRenyi(N, seed=42)
    x = hp_hp_sample(T, G, f_h, lambda_h)
    plot_joint_spectrum(x, G, "hp_hp")
    x = lp_lp_sample(T, G, f_l, lambda_l)
    plot_joint_spectrum(x, G, "lp_lp")
    x = hp_lp_sample(T, G, f_h, lambda_l)
    plot_joint_spectrum(x, G, "hp_lp")
    x = lp_hp_sample(T, G, f_l, lambda_h)
    plot_joint_spectrum(x, G, "lp_hp")
