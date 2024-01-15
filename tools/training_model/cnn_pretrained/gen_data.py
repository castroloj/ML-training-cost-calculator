from itertools import product
from random import shuffle

import keras.applications as tka
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.models import Sequential
from tqdm.auto import tqdm

from tools.constant import CNN_PRETRAINED_CONFIG
from tools.training_model.util.time_his import TimeHistoryBasic

import subprocess
import json
import sys


class GenData:
    def __init__(
        self,
        batch_sizes=CNN_PRETRAINED_CONFIG["batch_sizes"],
        optimizers=CNN_PRETRAINED_CONFIG["optimizers"],
        losses=CNN_PRETRAINED_CONFIG["losses"],
        epochs=CNN_PRETRAINED_CONFIG["epochs"],
        truncate_from=CNN_PRETRAINED_CONFIG["truncate_from"],
        trials=CNN_PRETRAINED_CONFIG["trials"],
    ):
        self.batch_sizes = batch_sizes
        self.epochs = epochs
        self.truncate_from = truncate_from
        self.trials = trials
        self.optimizers = optimizers
        self.losses = losses

    @staticmethod
    def nothing(x, **kwargs):
        return x

    # @staticmethod
    # def get_model(model_name, classes=None, input_shape=None):
    #     if classes is None:
    #         classes = 1000
    #     cls_model_method = getattr(tka, model_name)
    #     temp_model = cls_model_method()
    #     input_shape_default = temp_model.get_config()["layers"][0]["config"][
    #         "batch_input_shape"
    #     ][1:]
    #     if input_shape is None and classes == 1000:
    #         model = cls_model_method()
    #     elif input_shape is None:
    #         model = cls_model_method(
    #             include_top=False, input_shape=input_shape_default, classes=classes
    #         )
    #         model = Sequential([model, Flatten(), Dense(1000), Dense(classes)])
    #     else:
    #         model = cls_model_method(
    #             include_top=False, input_shape=tuple(input_shape), classes=classes
    #         )
    #         model = Sequential([model, Flatten(), Dense(1000), Dense(classes)])
    #     return model

    def get_train_data(
        self, num_data, model_name, input_shape=None, output_size=1000, progress=True, file_name=None
    ):
        model_data = []
        if progress:
            loop_fun = tqdm
        else:
            loop_fun = GenData.nothing

        # pick random combination
        shuffle(self.batch_sizes)
        shuffle(self.optimizers)
        shuffle(self.losses)
        comb = product(self.batch_sizes, self.optimizers, self.losses)
        model_configs = []
        # open file in write mode
        #with open(r'test.txt', 'w') as fp:
        if file_name:
            open(file_name, "w").close()
        for _ in loop_fun(range(num_data)):
            batch_size, optimizer, loss = next(comb)
            try:
                # Serialize the data structure to JSON
                params_json = json.dumps({
                    "model_name": model_name,
                    "batch_size": batch_size,
                    "output_size": output_size,
                    "optimizer": optimizer,
                    "loss": loss,
                    "epochs" : self.epochs,
                    "input_shape": input_shape,
                    "trials": self.trials,
                })
                result = subprocess.run(
                    ["python", "training_cnn_pretrained_subprocess.py", params_json],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    # stderr=sys.stderr, stdout=sys.stdout, # For debugging purposes
                )
                # Parse the JSON output from the subprocess
                result_values = json.loads(result.stdout)
                batch_size_data_batch = result_values["batch_size_data_batch"]
                batch_size_data_epoch = result_values["batch_size_data_epoch"]

                batch_times_truncated = batch_size_data_batch[self.truncate_from :]
                epoch_times_truncated = batch_size_data_epoch[self.truncate_from :]
                recovered_time = [
                    np.median(batch_times_truncated)
                ] * self.truncate_from + batch_times_truncated

                data_point = {
                    "batch_size": batch_size,
                    "batch_time_ms": np.median(batch_times_truncated),
                    "epoch_time_ms": np.median(epoch_times_truncated),
                    "setup_time_ms": np.sum(batch_size_data_batch) - sum(recovered_time),
                    "input_dim": input_shape,
                    "optimizer": optimizer,
                    "loss": loss,
                }
                model_data.append(data_point)
            except subprocess.CalledProcessError as e:
                print(f"Subprocess failed with return code {e.returncode}")
                # Uncomment for debbuging
                # print(f"Subprocess failed: {e}")
                # print("Standard Output:")
                # print(e.stdout)
                # print("Standard Error:")
                # print(e.stderr)
                # Handle the failure gracefully
                data_point = {
                    "batch_size": batch_size,
                    "batch_time_ms": None,
                    "epoch_time_ms": None,
                    "setup_time_ms": None,
                    "input_dim": input_shape,
                    "optimizer": optimizer,
                    "loss": loss,
                }
                model_data.append(data_point)
            if file_name:
                with open(file_name, 'a') as fp:
                    fp.write(f"{data_point}\n")
            model_configs.append({"optimizer": optimizer, "loss": loss})
        return model_data, model_configs

    @staticmethod
    def convert_train_data_to_dataframe(model_data):
        batch_sizes = [i['batch_size'] for i in model_data[0]]
        optimizers = [i['optimizer'] for i in model_data[0]]
        losses = [i['loss'] for i in model_data[0]]
        input_dims = [i['input_dim'] for i in model_data[0]]
        batch_time_ms = [i['batch_time_ms'] for i in model_data[0]]
        epoch_time_ms = [i['epoch_time_ms'] for i in model_data[0]]
        setup_time_ms = [i['setup_time_ms'] for i in model_data[0]]
        import pandas as pd
        model_data_df = pd.DataFrame(list(zip(batch_sizes, optimizers, losses, input_dims, batch_time_ms, epoch_time_ms, setup_time_ms)), columns = ['batch_size', 'optimizer', 'loss', 'input_dim', 'batch_time_ms', 'epoch_time_ms', 'setup_time_ms'])
        return model_data_df
