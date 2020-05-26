#################################
# This has to be added to find the custom packages
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../../')
#################################
import unittest
from unittest import TestCase

import logging
from ml.neural_net.digit_recog_1_layer import DigitNeuralNet1HiddenLayer
from ml.framework.file_utils import FileUtils

class TestDigitRecog1Layer(TestCase):

    def setUp(self):
        logging.info("Setup TestDigitRecog1Layer!!")

    def tearDown(self):
        logging.info("TearDown TestDigitRecog1Layer!!")

    def test_image_recog_two(self):
        img_path = FileUtils.path('images', 'two.png')

        net = DigitNeuralNet1HiddenLayer(784, 100, 10)
        net.load()
        q = net.predict(img_path)
        self.assertEqual(q, 2)

    def test_image_recog_four(self):
        img_path = FileUtils.path('images', 'four.png')

        net = DigitNeuralNet1HiddenLayer(784, 100, 10)
        net.load()
        q = net.predict(img_path)
        self.assertEqual(q, 4)

    def test_image_recog_five(self):
        img_path = FileUtils.path('images', 'five.png')

        net = DigitNeuralNet1HiddenLayer(784, 100, 10)
        net.load()
        q = net.predict(img_path)
        self.assertEqual(q, 5)

    def test_image_recog_eight(self):
        img_path = FileUtils.path('images', 'eight.png')

        net = DigitNeuralNet1HiddenLayer(784, 100, 10)
        net.load()
        q = net.predict(img_path)
        self.assertEqual(q, 3)




if __name__ == '__main__':
    unittest.main()
