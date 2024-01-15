from tools.training_model.cnn_pretrained.gen_data import GenData
from tools.hardware.gpu_feature import GpuFeatures
import pandas as pd
from datetime import datetime

c = datetime.now()
today = c.strftime('%y-%m-%d-%H-%M-%S')

# Get GPU description
gpu_features = GpuFeatures()
gpu_df = pd.DataFrame.from_dict(gpu_features.get_features(), orient='index').T

batch_sizes=[16, 32]
optimizers= ["sgd",
  "rmsprop",
  "adam",
  "adadelta",
  "adagrad",
  "adamax",
  "nadam",
  "ftrl",
]
losses=["mae",
  "mape",
  "mse",
  "msle",
  "poisson",
  "categorical_crossentropy",
  ]
epochs=10
truncate_from=2
trials=5
num_data = 20
MAX_LAYER_NUM = 30
# Generate model configurations

# train generated model configurations to get training time
mtd = GenData(optimizers=optimizers,losses=losses, epochs=epochs, truncate_from=truncate_from, trials=trials, batch_sizes=batch_sizes)

print('Start finished....')
model_data = mtd.get_train_data(num_data=num_data,  model_name='VGG16', file_name='test_output.txt')
print('Training finished....')

print(model_data)

batch_sizes = [i['batch_size'] for i in model_data[0]]
optimizers = [i['optimizer'] for i in model_data[1]]
losses = [i['loss'] for i in model_data[1]]
batch_time_ms = [i['batch_time_ms'] for i in model_data[0]]
epoch_time_ms = [i['epoch_time_ms'] for i in model_data[0]]
setup_time_ms = [i['setup_time_ms'] for i in model_data[0]]

model_data_df = pd.DataFrame(list(zip(batch_sizes, optimizers, losses, batch_time_ms, epoch_time_ms, setup_time_ms)), columns = ['batch_size', 'optimizer', 'loss', 'batch_time_ms', 'epoch_time_ms', 'setup_time_ms'])
print(model_data_df)
