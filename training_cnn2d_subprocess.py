# training_subprocess.py
from tools.training_model.util.time_his import TimeHistoryBasic
from tensorflow.python.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.python.keras.models import Sequential
import tensorflow as tf
import numpy as np
import sys
import json
import argparse

def build_cnn2d_model(kwargs_list, layer_orders):
    cnn2d = Sequential()
    for i, lo in enumerate(layer_orders):
        kwargs = kwargs_list[i]
        if lo == "Dense":
            cnn2d.add(Dense(**kwargs))
        elif lo == "Conv2D":
            cnn2d.add(Conv2D(**kwargs))
        elif lo == "MaxPooling2D":
            cnn2d.add(MaxPooling2D(**kwargs))
        elif lo == "Dropout":
            cnn2d.add(Dropout(**kwargs))
        elif lo == "Flatten":
            cnn2d.add(Flatten())
    kwargs = kwargs_list[-1]
    cnn2d.compile(metrics=["accuracy"], **kwargs["Compile"])
    return cnn2d

def main():
    if len(sys.argv) < 2:
        print("Usage: python training_subprocess.py <params_json>")
        sys.exit(1)
    #parser = argparse.ArgumentParser(description="Training Subprocess")
    #parser.add_argument("params_json", help="JSON containing training parameters")
    #args = parser.parse_args()


    # Parse the JSON argument
    params_json = sys.argv[1]
    #params_json = args.params_json
    params = json.loads(params_json)
    model_config_list = params['model_config_list']
    batch_size = params['batch_size']
    trials = params['trials']
    epochs = params['epochs']
    verbose = params['verbose']

    try:
        with tf.compat.v1.Session() as sess:
            kwargs_list = model_config_list[0]
            layer_orders = model_config_list[1]
            input_shape = model_config_list[2]
            model = build_cnn2d_model(kwargs_list, layer_orders)
            batch_size_data_batch = []
            batch_size_data_epoch = []
            out_shape = model.get_config()["layers"][-1]["config"]["units"]
            x = np.ones((batch_size, *input_shape), dtype=np.float32)
            y = np.ones((batch_size, out_shape), dtype=np.float32)
            for _ in range(trials):
                time_callback = TimeHistoryBasic()
                model.fit(
                    x,
                    y,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[time_callback],
                    verbose=verbose,
                )
                times_batch = np.array(time_callback.batch_times) * 1000
                times_epoch = np.array(time_callback.epoch_times) * 1000
                batch_size_data_batch.extend(times_batch)
                batch_size_data_epoch.extend(times_epoch)
        sess.close()
        # Print the values you want to return
        result_values = {
            "batch_size": batch_size,
            "input_shape": input_shape,
            "batch_size_data_batch": batch_size_data_batch,
            "batch_size_data_epoch": batch_size_data_epoch
        }
        print(json.dumps(result_values))
    except tf.errors.OpError as e:
        print(f"TensorFlow OpError during training: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"ValueError during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()