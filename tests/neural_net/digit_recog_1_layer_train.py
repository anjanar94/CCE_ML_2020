#################################
# This has to be added to find the custom packages
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../../')
#################################

import zipfile
from ml.neural_net.digit_recog_1_layer import DigitNeuralNet1HiddenLayer
from ml.framework.file_utils import FileUtils

#******************************************************************************#
#******************************************************************************#
train_data_zip_path = FileUtils.path('', 'mnist_train.csv.zip')
directory_to_extract_to = FileUtils.path('extra', '')

with zipfile.ZipFile(train_data_zip_path, 'r') as zip_ref:
    zip_ref.extractall(directory_to_extract_to)

#Neural Net Training
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.1
epoch = 5

train_data_path = FileUtils.path('extra', 'mnist_train.csv')

net = DigitNeuralNet1HiddenLayer(input_nodes, hidden_nodes, output_nodes)
net.train(train_data_path, epoch, learning_rate)
net.save()
