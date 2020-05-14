#################################
# This has to be added to find the custom packages
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../')
#################################
import unittest
from unittest import TestCase

import logging
import numpy
import pandas as pd

from ml.framework.file_utils import FileUtils
from ml.linear_classification import linearClassifier
from ml.linear_classification import LogisticRegression


class TestLinearRegression(TestCase):

    def setUp(self):
        logging.info("Setup TestLinearRegression!!")

    def tearDown(self):
        logging.info("TearDown TestLinearRegression!!")

    def test_linear_regression_model(self):
        train_file = FileUtils.path('', 'iris2.csv')
        test_file = FileUtils.path('', 'test_iris.csv')

        iris = pd.read_csv(train_file)
        test_iris = pd.read_csv(test_file)

        instanceOfLR, summary = linearClassifier(iris, test_iris, 4, 0.05, 200)
        instanceOfLR1, summary1 = linearClassifier(iris, test_iris, 4, 0.01, 500)
        instanceOfLR2, summary2 = linearClassifier(test_iris, iris, 4, 0.1, 100)

        print(instanceOfLR)
        print(summary)
        x = numpy.array([5.2, 3.5, 1.5, 0.2])
        print(instanceOfLR.predict(x))
        print(instanceOfLR1.numberOfInstance())


        print(instanceOfLR1)
        print(summary1)
        x = numpy.array([5.2, 3.5, 1.4, 0.2])
        print(instanceOfLR1.predict(x))
        print(instanceOfLR1.numberOfInstance())


        print(instanceOfLR2)
        print(summary2)
        x = numpy.array([5.2, 3.5, 1.6, 0.2])
        print(instanceOfLR2.predict(x))
        print(instanceOfLR2.numberOfInstance())

        #x = numpy.array([5.2, 3.5, 1.5, 0.2])
        #prediction = predict(x)
        #self.assertEqual(prediction, 'setosa')

        #x = numpy.array([5.9,3.0,4.2,1.5])
        #prediction = predict(x)
        #self.assertEqual(prediction, 'versicolor')

        #x = numpy.array([6.4,2.7,5.3,1.9])
        #c = predict(x)
        #self.assertEqual(prediction, 'versicolor')

if __name__ == '__main__':
    unittest.main()
