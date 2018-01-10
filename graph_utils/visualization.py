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
    plt.imshow(20 * np.log10(np.abs(xflambda)), origin="lower", extent=[-T // 2, T // 2, 0, N - 1], aspect=T/N)
    plt.colorbar()
    plt.title("JFT(x)")
    plt.xlabel("f")
    plt.ylabel("lambda")
    fig.savefig(file + ".png")


def plot_temporal_matrix(x, file):
    fig = plt.figure()
    N, T = x.shape
    plt.imshow(x, origin="lower", aspect=T/N)
    plt.colorbar()
    plt.title("Signal x")
    plt.xlabel("t")
    plt.ylabel("v")
    fig.savefig(file + ".png")


def plot_tf_fir_filter(sess, filter, file=None):
    K, M, C, F = filter.get_shape()
    F = int(F)
    C = int(C)
    fig = plt.figure()
    weights = sess.run(filter)
    for n in range(F):
        plt.subplot(F // 8 + 1, 8, n+1)
        plot_tv_fir_frequency_response(weights[:, :, np.random.randint(C), n], with_axis=False)

    if file is None:
        fig.savefig(filter.name.replace("/", "_") + ".png")
    else:
        fig.savefig(file + ".png")


def plot_tv_fir_frequency_response(h, file=None, N=200, kernel="chebyshev", with_axis=True):
    K, M = h.shape
    lambdas = np.linspace(-1, 1, N)
    w = np.linspace(-np.pi, np.pi, N)
    wv, lv = np.meshgrid(w, lambdas)
    ejwv = np.exp(1j * wv)
    H = np.zeros((N, N), dtype=np.complex128)
    for k in range(K):
        for m in range(M):
            if kernel == "chebyshev":
                H += h[k, m] * (ejwv ** m) * cheb_poly(lv, k)

    if file is not None:
        fig = plt.figure()
    plt.imshow(np.abs(H), origin="lower", extent=[-np.pi, np.pi, -1, 1], aspect=np.pi)
    if with_axis:
        plt.colorbar()
        plt.title("TV-FIR joint frequency response")
        plt.xlabel("$\omega$")
        plt.ylabel("$\lambda$")
    else:
        plt.axis("off")

    if file is not None:
        fig.savefig(file + ".png")


def plot_chebyshev_frequency_response(h, file=None, N=200, with_axis=True):
    K = len(h)
    lambdas = np.linspace(-1, 1, N)
    H = np.zeros(N, dtype=np.float32)
    for k in range(K):
        H += h[k] * cheb_poly(lambdas, k)

    if file is not None:
        fig = plt.figure()

    plt.plot(lambdas, np.abs(H))

    if with_axis:
        plt.title("Chebyshev filter frequency response")
        plt.xlabel("$\lambda$")
    else:
        plt.axis("off")

    if file is not None:
        fig.savefig(file + ".png")


def cheb_poly(l, K):
    T_0 = 1
    T = T_0

    if K > 0:
        T_1 = l
        T = T_1

    for k in range(2, K+1):
        T_2 = 2 * T_1 * l - T_0
        T_0 = T_1
        T_1 = T_2
        T = T_2

    return T


if __name__ == '__main__':
    from pygsp import graphs
    from synthetic_data.data_generation import generate_spectral_samples_hard

    N = 100
    T = 128
    f_h = 50
    f_l = 15
    lambda_h = 80
    lambda_l = 15
    # G = graphs.Community(N)
    # x, _ = generate_spectral_samples_hard(1, T, G, f_h, lambda_h, f_l, lambda_l, sigma=2, sigma_n=1)
    # # x = hp_hp_sample(T, G, f_h, lambda_h)
    # plot_joint_spectrum(x[0, :, :, 0], G, "hp_hp")
    # plot_temporal_matrix(x[0, :, :, 0], "hp_hp_t")
    # # x = lp_lp_sample(T, G, f_l, lambda_l)
    # plot_joint_spectrum(x[1, :, :, 0], G, "lp_lp")
    # plot_temporal_matrix(x[1, :, :, 0], "lp_lp_t")
    # # x = hp_lp_sample(T, G, f_h, lambda_l)
    # plot_joint_spectrum(x[2, :, :, 0], G, "lp_hp")
    # plot_temporal_matrix(x[2, :, :, 0], "lp_hp_t")
    # # x = lp_hp_sample(T, G, f_l, lambda_h)
    # plot_joint_spectrum(x[3, :, :, 0], G, "hp_lp")
    # plot_temporal_matrix(x[3, :, :, 0], "hp_lp_t")
    # # Combinations
    # plot_joint_spectrum(x[4, :, :, 0], G, "lp_hp_hp_lp")
    # plot_temporal_matrix(x[4, :, :, 0], "lp_hp_hp_lp_t")
    # # x = lp_hp_sample(T, G, f_l, lambda_h)
    # plot_joint_spectrum(x[5, :, :, 0], G, "hp_hp_lp_lp")
    # plot_temporal_matrix(x[5, :, :, 0], "hp_hp_lp_lp_t")
    # plot_tv_fir_frequency_response(np.random.randn(3, 3), file="seaborn")

    fig = plt.figure()
    for n in range(100):
        plt.subplot(10, 10, n+1)
        plot_tv_fir_frequency_response(np.random.randn(3, 6), with_axis=False)

    fig.savefig("many.png")