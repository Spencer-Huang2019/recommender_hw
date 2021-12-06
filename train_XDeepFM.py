import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import scrapbook as sb
from tempfile import TemporaryDirectory
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

from utils.constants import SEED
from utils.deeprec_utils import (
    download_deeprec_resources, prepare_hparams
)
from xDeepFM import XDeepFMModel
from utils.iterator import FFMTextIterator

print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))

# Parameters
EPOCHS_FOR_SYNTHETIC_RUN = 15
EPOCHS_FOR_CRITEO_RUN = 10
BATCH_SIZE_SYNTHETIC = 128
BATCH_SIZE_CRITEO = 4096
RANDOM_SEED = SEED  # Set to None for non-deterministic result

data_path = "data/xdeepfmresources"
yaml_file = os.path.join(data_path, r'xDeepFM.yaml')
train_file = os.path.join(data_path, r'synthetic_part_0')
valid_file = os.path.join(data_path, r'synthetic_part_1')
test_file = os.path.join(data_path, r'synthetic_part_2')
output_file = os.path.join(data_path, r'output.txt')

if not os.path.exists(yaml_file):
    download_deeprec_resources(r'https://recodatasets.z20.web.core.windows.net/deeprec/', data_path,
                               'xdeepfmresources.zip')


# 利用synthetic数据集进行训练和验证
def synthetic():
    # prepare hyper-parameters
    hparams = prepare_hparams(yaml_file,
                              FEATURE_COUNT=1000,
                              FIELD_COUNT=10,
                              cross_l2=0.0001,
                              embed_l2=0.0001,
                              learning_rate=0.001,
                              epochs=EPOCHS_FOR_SYNTHETIC_RUN,
                              batch_size=BATCH_SIZE_SYNTHETIC)
    # print(hparams)

    # 为模型创建一个迭代器
    input_creator = FFMTextIterator

    # create model
    model = XDeepFMModel(hparams, input_creator, seed=RANDOM_SEED)
    # print(model.run_eval(test_file))

    # train model
    model.fit(train_file, valid_file)

    # evaluate model
    res_syn = model.run_eval(test_file)
    print(res_syn)

    sb.glue("res_syn", res_syn)

    model.predict(test_file, output_file)


# 利用criteo数据集进行训练和验证
def criteo():
    # hyper parameters
    hparams = prepare_hparams(yaml_file,
                              FEATURE_COUNT=2300000,
                              FIELD_COUNT=39,
                              cross_l2=0.01,
                              embed_l2=0.01,
                              layer_l2=0.01,
                              learning_rate=0.002,
                              batch_size=BATCH_SIZE_CRITEO,
                              epochs=EPOCHS_FOR_CRITEO_RUN,
                              cross_layer_sizes=[20, 10],
                              init_value=0.1,
                              layer_sizes=[20, 20],
                              use_Linear_part=True,
                              use_CIN_part=True,
                              use_DNN_part=True)

    train_file = os.path.join(data_path, r'cretio_tiny_train')
    valid_file = os.path.join(data_path, r'cretio_tiny_valid')
    test_file = os.path.join(data_path, r'cretio_tiny_test')

    # designate a data iterator for the model
    input_creator = FFMTextIterator

    # create model
    model = XDeepFMModel(hparams, input_creator, seed=RANDOM_SEED)
    # print(model.run_eval(test_file))

    # train model
    model.fit(train_file, valid_file)

    # evaluate model
    res_syn = model.run_eval(test_file)
    print(res_syn)

    sb.glue("res_syn", res_syn)

    model.predict(test_file, output_file)


if __name__ == '__main__':
    # synthetic()
    criteo()