import os
import pandas as pd
import json
import seaborn as sns

EXPERIMENTS_DIR = "signal_classification/experiments"
BATCHES_DIR = os.path.join(os.curdir, EXPERIMENTS_DIR, "different_dropout_1200")
CSV_DIR = os.path.join(os.curdir, EXPERIMENTS_DIR)

BATCHES = [0, 1, 2, 3, 4, 5]


final = pd.DataFrame(columns=["shot_noise", "type", "test", "train"])


for batch_dir in os.listdir(BATCHES_DIR):
    dict = {}
    num_batch = (batch_dir.split("_")[1])
    batch_dir = os.path.join(BATCHES_DIR, batch_dir)
    if int(num_batch) not in BATCHES:
        continue
    if os.path.isdir(batch_dir):
        for ntw_dir in os.listdir(batch_dir):
            dict["type"] = [ntw_dir]
            ntw_dir = os.path.join(batch_dir, ntw_dir)
            if os.path.isdir(ntw_dir):
                for exp_dir in os.listdir(ntw_dir):
                    exp_dir = os.path.join(ntw_dir, exp_dir)
                    if os.path.isdir(ntw_dir):
                        if not os.path.exists(os.path.join(exp_dir, "params.json")):
                            continue
                        with open(os.path.join(exp_dir, "params.json")) as f:
                            params = json.load(f)
                        dict["shot_noise"] = [params["shot_noise"]]
                        for phase in ["test", "train"]:
                            test_path = os.path.join(exp_dir, phase)
                            #os.system("python exportTensorFlowLog.py " + test_path + " " + test_path + " scalars")
                            df = pd.read_csv(test_path + "/scalars.csv")
                            if phase == "test":
                                print(df.tail(1).iloc[0])
                                test_accuracy = df.tail(1).iloc[0]["test_accuracy"]
                                dict["test"] = [test_accuracy]
                            else:
                                train_accuracy = df.tail(1).iloc[0]["metric/accuracy_1"]
                                dict["train"] = [train_accuracy]
                                print(dict["train"])
                        final = final.append(pd.DataFrame(dict), ignore_index=True)
                        print(final)

final.to_csv(os.path.join(CSV_DIR, 'data_frame_sizes.csv'), sep=",")
print(final.pivot_table(index="shot_noise", columns="type", values="test"))
sns.set(style="whitegrid", color_codes=True)
sns_plot = sns.factorplot(x="shot_noise", y="test", hue="type", data=final, kind="bar")
sns_plot.savefig(CSV_DIR + "/plot.png")