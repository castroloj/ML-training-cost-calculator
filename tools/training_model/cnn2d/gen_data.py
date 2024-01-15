from random import sample
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

from tools.constant import CNN2D_CONFIG
from tools.constant import TIME_COLUMNS
from tools.training_model.cnn2d.gen_model import GenModel
from tools.training_model.util.time_his import TimeHistoryBasic

import subprocess
import json
import sys

class GenData:
    def __init__(
        self,
        model_configs,
        batch_sizes=CNN2D_CONFIG["batch_sizes"],
        epochs=CNN2D_CONFIG["epochs"],
        truncate_from=CNN2D_CONFIG["truncate_from"],
        trials=CNN2D_CONFIG["trials"],
        activation_fcts=CNN2D_CONFIG["activation_fcts"],
        optimizers=CNN2D_CONFIG["optimizers"],
        losses=CNN2D_CONFIG["losses"],
        paddings=CNN2D_CONFIG["paddings"],
    ):
        self.model_configs = []
        for info_list in model_configs:
            self.model_configs.append(info_list.copy())
        self.batch_sizes = batch_sizes
        self.epochs = epochs
        self.truncate_from = truncate_from
        self.trials = trials
        self.activation_fcts = activation_fcts
        self.optimizers = optimizers
        self.losses = losses
        self.paddings = paddings

    def get_train_data(self, progress=True, verbose=False, file_name=None):
        model_data = []
        if progress:
            loop_fun = tqdm
        else:
            loop_fun = GenModel.nothing
        #for info_list in self.model_configs:
        #    model_configs.append(info_list.copy())
        model_configs = [info_list.copy() for info_list in self.model_configs]            
        # open file in write mode
        #with open(r'test.txt', 'w') as fp:
        if file_name:
            open(file_name, "w").close()
        for model_config_list in loop_fun(model_configs):
            batch_size = sample(self.batch_sizes, 1)[0]
            input_shape = model_config_list[2]
            try:
                # Serialize the data structure to JSON
                params_json = json.dumps({
                    "model_config_list": model_config_list,
                    "batch_size": batch_size,
                    "trials" : self.trials,
                    "epochs" : self.epochs,
                    "verbose" : verbose
                })
                result = subprocess.run(
                    ["python", "training_cnn2d_subprocess.py", params_json],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    #stderr=sys.stderr, stdout=sys.stdout, # For debugging purposes
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
                
                model_config_list.append(
                    {
                        "batch_size": batch_size,
                        "batch_time_ms": np.median(batch_times_truncated),
                        "epoch_time_ms": np.median(epoch_times_truncated),
                        "setup_time_ms": np.sum(batch_size_data_batch)
                        - sum(recovered_time),
                        "input_dim": input_shape,
                    }
                )
            except subprocess.CalledProcessError as e:
                print(f"Subprocess failed with return code {e.returncode}")
                # Uncomment for debbuging
                # print(f"Subprocess failed: {e}")
                # print("Standard Output:")
                # print(e.stdout)
                # print("Standard Error:")
                # print(e.stderr)
                # Handle the failure gracefully
                model_config_list.append(
                    {
                        "batch_size": batch_size,
                        "batch_time_ms": None,
                        "epoch_time_ms": None,
                        "setup_time_ms": None,
                        "input_dim": input_shape,
                    }
                )
            model_data.append(model_config_list)
            if file_name:
                with open(file_name, 'a') as fp:
                    fp.write(f"{model_config_list}\n")
        return model_data
    
    def convert_config_data(
        self,
        model_data,
        max_layer_num=105,
        num_fill_na=0,
        name_fill_na=None,
        min_max_scaler=True,
    ) -> Tuple[List[pd.DataFrame], pd.DataFrame, Optional[MinMaxScaler]]:

        feature_columns = [
            "layer_type",
            "layer_size",
            "kernel_size",
            "strides",
            "padding",
            "activation",
            "optimizer",
            "loss",
            "batch_size",
            "input_shape",
            "channels",
        ]
        time_columns = TIME_COLUMNS
        feature_layer_types = ["Conv2D", "MaxPooling2D", "Dense"]

        row_num = max(
            [
                len(self.activation_fcts),
                len(self.optimizers),
                len(self.losses),
                len(self.paddings),
                len(feature_layer_types),
            ]
        )
        pos_dict = dict((i, feature_columns.index(i)) for i in feature_columns)
        values_dict = {
            "activation": self.activation_fcts,
            "optimizer": self.optimizers,
            "loss": self.losses,
            "padding": self.paddings,
            "layer_type": feature_layer_types,
        }
        empty_rows = [[None] * len(feature_columns)] * row_num
        empty_rows = [i[:] for i in empty_rows]  # break connection for lists
        for v_type, v_list in values_dict.items():
            for index, value in enumerate(v_list):
                empty_rows[index][pos_dict[v_type]] = value

        model_data_dfs = []
        time_rows = []
        for model_info in tqdm(model_data):
            data_rows = []
            kwargs_list = model_info[0]
            layer_orders = model_info[1]
            input_shape = model_info[2][0]
            channels = model_info[2][-1]
            train_times = model_info[3]
            for index, layer_type in enumerate(layer_orders):
                values = kwargs_list[index]
                if layer_type == "Conv2D":
                    data_rows.append(
                        [
                            layer_type,
                            values["filters"],
                            values["kernel_size"][0],
                            values["strides"][0],
                            values["padding"],
                            values["activation"],
                            kwargs_list[-1]["Compile"]["optimizer"],
                            kwargs_list[-1]["Compile"]["loss"],
                            train_times["batch_size"],
                            input_shape,
                            channels,
                        ]
                    )
                elif layer_type == "MaxPooling2D":
                    data_rows.append(
                        [
                            layer_type,
                            num_fill_na,
                            values["pool_size"][0],
                            values["strides"][0],
                            values["padding"],
                            name_fill_na,
                            kwargs_list[-1]["Compile"]["optimizer"],
                            kwargs_list[-1]["Compile"]["loss"],
                            train_times["batch_size"],
                            input_shape,
                            channels,
                        ]
                    )
                elif layer_type == "Dense":
                    data_rows.append(
                        [
                            layer_type,
                            values["units"],
                            num_fill_na,
                            num_fill_na,
                            name_fill_na,
                            values["activation"],
                            kwargs_list[-1]["Compile"]["optimizer"],
                            kwargs_list[-1]["Compile"]["loss"],
                            train_times["batch_size"],
                            input_shape,
                            channels,
                        ]
                    )
                else:
                    pass
            time_rows.append(
                [
                    train_times["batch_time_ms"],
                    train_times["epoch_time_ms"],
                    train_times["setup_time_ms"],
                ]
            )
            data_rows.extend(empty_rows)
            temp_df = pd.DataFrame(data_rows, columns=feature_columns)

            temp_df = pd.get_dummies(temp_df)
            temp_df = temp_df.drop(temp_df.index.tolist()[-len(empty_rows) :])

            columns_count = len(temp_df.columns)
            zero_rows = np.zeros((max_layer_num, columns_count))
            temp_array = temp_df.to_numpy()
            temp_array = np.append(temp_array, zero_rows, 0)
            temp_array = temp_array[
                :max_layer_num,
            ]
            temp_df = pd.DataFrame(temp_array, columns=temp_df.columns)
            model_data_dfs.append(temp_df)
        time_df = pd.DataFrame(time_rows, columns=time_columns)
        if min_max_scaler:
            scaled_model_dfs = []
            scaler = MinMaxScaler()
            scaler.fit(pd.concat(model_data_dfs, axis=0).to_numpy())
            for data_df in model_data_dfs:
                scaled_data = scaler.transform(data_df.to_numpy())
                scaled_temp_df = pd.DataFrame(scaled_data, columns=temp_df.columns)
                scaled_model_dfs.append(scaled_temp_df)
            return scaled_model_dfs, time_df, scaler
        return model_data_dfs, time_df, None

    @staticmethod
    def model_data_to_dataframe(model_data, total_layers=29):
        data = {
            'batch_size': [],
            'input_shape': [],
            'optimizer': [],
            'loss': [],
            'channels': [],
            'batch_time_ms': [],
            'epoch_time_ms': [],
            'setup_time_ms': [],
        }

        for i in range(total_layers):
            data[f'layer_size_{i}'] = []
            data[f'layer_type_{i}'] = []
            data[f'activation_type_{i}'] = []
            data[f'kernel_size_{i}'] = []
            data[f'padding_{i}'] = []
            data[f'strides_{i}'] = []

        # Iterate over models
        for row in model_data:
            data['batch_size'].append(row[3]['batch_size'])
            data['input_shape'].append(row[2][0])# data['input_shape'].append(row[2])
            data['optimizer'].append(row[0][-1]["Compile"]["optimizer"])
            data['loss'].append(row[0][-1]["Compile"]["loss"])
            data['channels'].append(row[2][-1])
            data['batch_time_ms'].append(row[3]["batch_time_ms"])
            data['epoch_time_ms'].append(row[3]["epoch_time_ms"])
            data['setup_time_ms'].append(row[3]["setup_time_ms"])

            for i in range(total_layers):
                if i < len(row[1]):
                    data[f'layer_type_{i}'].append(row[1][i])
                    if 'filters' in row[0][i]:
                        data[f'layer_size_{i}'].append(row[0][i]['filters'])
                    elif 'units' in row[0][i]:
                        data[f'layer_size_{i}'].append(row[0][i]['units'])
                    else:
                        data[f'layer_size_{i}'].append(0)
                    data[f'activation_type_{i}'].append(row[0][i].get('activation', None))
                    data[f'kernel_size_{i}'].append(row[0][i].get('kernel_size', 0))
                    data[f'padding_{i}'].append(row[0][i].get('padding', None))
                    if 'strides' in row[0][i]:
                        data[f'strides_{i}'].append(row[0][i]['strides'][0]*row[0][i]['strides'][1])
                    else:
                        data[f'strides_{i}'].append(0)
                else:
                    # If i is greater than or equal to the number of layers, append None
                    for suffix in ['layer_type', 'activation_type', 'padding']:
                        data[f'{suffix}_{i}'].append(None)
                    for suffix in ['layer_size', 'kernel_size', 'strides']:
                        data[f'{suffix}_{i}'].append(0)

        model_data_df = pd.DataFrame(data)
        return model_data_df
    
    @staticmethod
    def model_data_file_to_dataframe(file_path, total_layers=29):
        import ast
        parsed_data = []
        with open(file_path, 'r') as file:
            for line in file:
                # Remove leading and trailing whitespaces, and convert to Python object
                row_data = ast.literal_eval(line.strip())
                parsed_data.append(row_data)
        return GenData.model_data_to_dataframe(parsed_data, total_layers)