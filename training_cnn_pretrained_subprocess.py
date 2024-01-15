# training_subprocess.py
from tools.training_model.util.time_his import TimeHistoryBasic
import keras.applications as tka
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.models import Sequential
import tensorflow as tf

import numpy as np
import sys
import json
import argparse

def get_model(model_name, classes=None, input_shape=None):
    if classes is None:
        classes = 1000
    cls_model_method = getattr(tka, model_name)
    temp_model = cls_model_method()
    input_shape_default = temp_model.get_config()["layers"][0]["config"][
        "batch_input_shape"
    ][1:]
    if input_shape is None and classes == 1000:
        model = cls_model_method()
    elif input_shape is None:
        model = cls_model_method(
            include_top=False, input_shape=input_shape_default, classes=classes
        )
        model = Sequential([model, Flatten(), Dense(1000), Dense(classes)])
    else:
        model = cls_model_method(
            include_top=False, input_shape=tuple(input_shape), classes=classes
        )
        model = Sequential([model, Flatten(), Dense(1000), Dense(classes)])
    return model

def main():
    if len(sys.argv) < 2:
        print("Usage: python training_subprocess.py <params_json>")
        sys.exit(1)
    #parser = argparse.ArgumentParser(description="Training Subprocess")
    #parser.add_argument("params_json", help="JSON containing training parameters")
    #args = parser.parse_args()

    # Parse the JSON argument
    params_json = sys.argv[1]
    params = json.loads(params_json)
    model_name = params['model_name']
    batch_size = params['batch_size']
    output_size = params['output_size']
    optimizer = params['optimizer']
    loss = params['loss']
    epochs = params['epochs']
    input_shape = params['input_shape']
    trials = params["trials"]

    try:
        with tf.compat.v1.Session() as sess:
            gpu_devices = tf.config.experimental.list_physical_devices("GPU")
            # The following does not work on my environment
            # for device in gpu_devices:
            #     tf.config.experimental.set_memory_growth(device, True)
            # It is replaced by the folliwing
            for device in gpu_devices:
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                except:
                    print(f"Invalid device or cannot modify virtual devices once initialized: {device}.")

            model = get_model(
                model_name, classes=output_size, input_shape=input_shape
            )
            model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
            input_shape = model.get_config()["layers"][0]["config"][
                "batch_input_shape"
            ][1:]
            batch_size_data_batch = []
            batch_size_data_epoch = []
            x = np.ones((batch_size, *input_shape), dtype=np.float32)
            y = np.ones((batch_size, output_size), dtype=np.float32)
            for _ in range(trials):
                time_callback = TimeHistoryBasic()
                model.fit(
                    x,
                    y,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[time_callback],
                    verbose=False,
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