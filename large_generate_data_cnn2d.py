from tools.training_model.cnn2d.gen_data import GenData
from tools.training_model.cnn2d.gen_model import GenModel
from tools.training_model.cnn2d.gen_data_helpers.flop_level_data import FlopLevelData
from tools.hardware.gpu_feature import GpuFeatures
import pandas as pd
from datetime import datetime


# Tabular data
INPUT_SHAPE_LOWER=8
INPUT_SHAPE_UPPER=32
CONV_LAYER_NUM_LOWER=3
CONV_LAYER_NUM_UPPER=15
DENSE_LAYER_NUM_LOWER=1
DENSE_LAYER_NUM_UPPER=10
DENSE_SIZE_LOWER=16
DENSE_SIZE_UPPER=64
INPUT_CHANNELS=[3]
INPUT_SHAPE_LIST = [16, 32, 64, 128, 256]

BATCH_SIZES=[16, 32, 64]

MAX_LAYER_NUM = CONV_LAYER_NUM_UPPER + DENSE_LAYER_NUM_UPPER

c = datetime.now()
today = c.strftime('%y-%m-%d-%H-%M-%S')

# Get GPU description
gpu_features = GpuFeatures()
gpu_df = pd.DataFrame.from_dict(gpu_features.get_features(), orient='index').T

gpu_df.to_csv(f'data/gpu_specs/aipaca_gpu_specs_{today}.csv')

# Generate model configurations

# Set parameters for the random model generator
gen = GenModel(input_shape_lower=INPUT_SHAPE_LOWER,
               input_shape_upper=INPUT_SHAPE_UPPER,
               conv_layer_num_lower=CONV_LAYER_NUM_LOWER,
               conv_layer_num_upper=CONV_LAYER_NUM_UPPER,
               dense_layer_num_lower=DENSE_LAYER_NUM_LOWER,
               dense_layer_num_upper=DENSE_LAYER_NUM_UPPER,
               dense_size_lower=DENSE_SIZE_LOWER,
               dense_size_upper=DENSE_SIZE_UPPER,
               input_channels=INPUT_CHANNELS,
               input_shape_list=INPUT_SHAPE_LIST,
               )

# generate model configurations as data points
data_points = 2000
print(('Before models are generated'))
model_configs = gen.generate_model_configs(num_model_data=data_points)
print(('After models are generated'))
with open(f'data/model_config/model_configs_{today}.txt', 'w') as fp:
    for model_config in model_configs:
        fp.write(f"{model_config}\n")

# train generated model configurations to get training time
mtd = GenData(model_configs,
              batch_sizes=BATCH_SIZES)

print('Start finished....')
model_data = mtd.get_train_data(file_name=f'data/model_data/model_data_{today}.txt')
print('Training finished....')