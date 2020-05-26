#################################
# This has to be added to find the custom packages
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../../')
#################################
import unittest
from unittest import TestCase

import logging
import zipfile
from ml.neural_net.digit_recog_2_layer import DigitNeuralNet2HiddenLayer
from ml.framework.file_utils import FileUtils

class TestDigitRecog2Layer(TestCase):

    def setUp(self):
        logging.info("Setup TestDigitRecog2Layer!!")

    def tearDown(self):
        logging.info("TearDown TestDigitRecog2Layer!!")

    def test_model_testing(self):
        test_data_zip_path = FileUtils.path('', 'mnist_test.csv.zip')
        directory_to_extract_to = FileUtils.path('extra', '')

        with zipfile.ZipFile(test_data_zip_path, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)

        test_data_path = FileUtils.path('extra', 'mnist_test.csv')

        net = DigitNeuralNet2HiddenLayer(784, 100, 50, 10)
        net.load()

        confusion_matrix, accuray = net.test(test_data_path)
        #self.assertEqual(accuray, 13.79)

    def test_image_recog_two(self):
        img_path = FileUtils.path('images', 'two.png')

        net = DigitNeuralNet2HiddenLayer(784, 100, 50, 10)
        net.load()
        q = net.predict(img_path)
        #self.assertEqual(q, 5)

    def test_image_recog_four(self):
        img_path = FileUtils.path('images', 'four.png')

        net = DigitNeuralNet2HiddenLayer(784, 100, 50, 10)
        net.load()
        q = net.predict(img_path)
        #self.assertEqual(q, 2)

    def test_image_recog_five(self):
        img_path = FileUtils.path('images', 'five.png')

        net = DigitNeuralNet2HiddenLayer(784, 100, 50, 10)
        net.load()
        q = net.predict(img_path)
        #self.assertEqual(q, 5)

    def test_image_recog_eight(self):
        img_path = FileUtils.path('images', 'eight.png')

        net = DigitNeuralNet2HiddenLayer(784, 100, 50, 10)
        net.load()
        q = net.predict(img_path)
        #self.assertEqual(q, 3)

if __name__ == '__main__':
    unittest.main()
