import os
import tempfile

TEMPDIR = "/users/gortizjimenez/Developer/tv_graph_cnn/signal_classification/experiments/"


def _next_batch(log_dir):
    exp_numbers = []
    if not os.path.exists(log_dir):
        return 0
    for file in os.listdir(log_dir):
        if "batch" not in file:
            continue
        else:
            exp_numbers.append(int(file.split("_")[1]))
    return max(exp_numbers) if len(exp_numbers) > 0 else 0


if __name__ == '__main__':
    # Batch parameters
    params = {
        "--log_dir": os.path.join(TEMPDIR, "batch_"+str(_next_batch(TEMPDIR))),
        "--num_vertices": 100,
        "--num_frames": 128,
        "--num_classes": 6,
        "--num_train": 12000,
        "--num_test": 1200,
        "--sigma": 2,
        "--num_epochs": 10,
        "--learning_rate": 1e-4,
        "--batch_size": 100,
        "--vertex_filter_orders": [3, 3, 3],
        "--time_filter_orders": [3, 3, 3],
        "--num_filters": [8, 16, 32],
        "--time_poolings": [4, 4, 4],
        "--vertex_poolings": [2, 2, 2],
        "--f_h": 50,
        "--f_l": 15,
        "--lambda_h": 80,
        "--lambda_l": 15,
        "--action": "train"
    }

    models = ["deep_cheb", "deep_fir"]
    noises = [0.01, 0.1, 0.5, 1, 1.5, 2]

    for model in models:
        if model == "deep_cheb":
            params["--vertex_filter_orders"] = [4, 4, 4]
        else:
            params["--vertex_filter_orders"] = [3, 3, 3]

        for sigma_n in noises:
            args = []
            for arg_name, value in params.items():
                if isinstance(value, list):
                    args.append(arg_name + " " + " ".join(str(e) for e in value))
                else:
                    args.append(arg_name + " " + str(value))
            args.append("--model_type " + model)
            args.append("--sigma_n " + str(sigma_n))
            exit = 1
            print("Simulating %s with sigma_n %.2f" % (model, sigma_n))
            while exit != 0:
                exit = os.system("python signal_classification/test.py " + " ".join(args))


