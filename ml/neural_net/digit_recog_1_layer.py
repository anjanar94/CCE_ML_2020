import os
import numpy
from numpy import savetxt
from numpy import loadtxt
import json
import matplotlib.pyplot
from skimage.color import rgb2gray
from skimage.color import rgba2rgb
from skimage import io

from ml.neural_net.custom_neural_net import DigitNeuralNetI
from ml.framework.file_utils import FileUtils


class DigitNeuralNet1HiddenLayer(DigitNeuralNetI):

    def __init__(self,inp_nodes,hid_nodes,out_nodes):
        self.inodes = inp_nodes
        self.hnodes = hid_nodes
        self.onodes = out_nodes
        self.lr = 0.1
        self.epoch = 5
        self.activation_function = lambda x: 1/(1+pow(2.718,-x))
        self.wih = numpy.random.rand(self.hnodes,self.inodes)-0.5
        self.who = numpy.random.rand(self.onodes,self.hnodes)-0.5

        self.params = {}
        self.params['Hidden Layer'] = 1
        self.params['Total Layer'] = 3
        self.params['Input Nodes'] = self.inodes
        self.params['Hidden Nodes'] = self.hnodes
        self.params['Output Nodes'] = self.onodes
        self.params['Learning Rate'] = self.lr
        self.params['Activation Function'] = 'Binary Sigmoid'
        self.con_mat = {}
        pass

    def __train(self,input_list,target_list):
        inputs = numpy.array(input_list,ndmin = 2).T
        targets = numpy.array(target_list, ndmin = 2).T
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T,output_errors)

        self.who += self.lr * numpy.dot((output_errors*final_outputs*(1-final_outputs)),numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors*hidden_outputs*(1-hidden_outputs)),numpy.transpose(inputs))
        pass

    def __query(self,input_list):
        inputs = numpy.array(input_list,ndmin = 2).T
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    ###################################
    ### Implement Interface Methods ###
    ###################################
    def train(self, path: str, epoch: int, lr: float):
        self.lr = lr
        self.epoch = epoch
        self.params['Epoch'] = self.epoch
        self.params['Learning Rate'] = self.lr

        training_data_file = open(path,"r")
        training_data_list = training_data_file.readlines()
        training_data_file.close()
        self.params['Total Train Data Points'] = len(training_data_list)
        
        for e in range(epoch):
            print('### Epoch = ' + str(e))
            for records in training_data_list:
                all_values = records.split(',')
                inputs = (numpy.asfarray(all_values[1:])/255 * 0.99) + 0.01
                targets = numpy.zeros(self.onodes)+0.01
                targets[int(all_values[0])] = 0.99
                self.__train(inputs, targets)
        pass

    def test(self, path: str) -> ({}, float):
        d = {}
        total = []
        for clazz in range(10):
            d[clazz] = {'t_rel':0, 't_ret':0, 'rr':0}

        test_data_file = open(path,"r")
        test_data_list = test_data_file.readlines()
        test_data_file.close()

        for record in test_data_list:
            all_values = record.split(',')
            inputs = (numpy.asfarray(all_values[1:])/255*0.99)+0.01
            clazz = int(all_values[0])
            outputs = self.__query(inputs)
            prediction = numpy.argmax(outputs)

            d[clazz]['t_rel'] = d[clazz]['t_rel'] + 1
            d[prediction]['t_ret'] = d[prediction]['t_ret'] + 1
            if clazz == prediction:
                d[clazz]['rr'] = d[clazz]['rr'] + 1
                total.append(1)
            else:
                total.append(0)

        result_array = numpy.asarray(total)
        accuracy = (result_array.sum()/result_array.size)*100
        self.params['Total Test Data Points'] = len(total)
        self.params['Accuracy'] = accuracy
        self.con_mat = d
        print("Accuracy = ",result_array.sum()/result_array.size*100,"%",sep='')
        print("Confusion Matrix = ",self.con_mat,sep='')
        return (d, accuracy)

    def predict(self, img_path: str) -> int:
        original = io.imread(img_path)
        shape = str(original.shape).replace('(', '').replace(')', '')
        dim = int(shape.split(',')[2].strip())
        if dim > 3:
            grayscale = rgb2gray(rgba2rgb(original))
        else:
            grayscale = rgb2gray(original)

        img_array = grayscale.reshape(784,)
        img_array = 1-img_array
        query = self.__query(img_array)
        max = numpy.argmax(query)
        return max

    def load(self):
        dir = os.path.join('nets', 'digit_1_hidden_layer')
        params_file_path =  FileUtils.path(dir, 'params.json')
        con_mat_file_path =  FileUtils.path(dir, 'con_mat.json')
        wih_file_path =  FileUtils.path(dir, 'wih.csv')
        who_file_path =  FileUtils.path(dir, 'who.csv')

        out_param_file = open(params_file_path)
        self.params = json.load(out_param_file)
        out_param_file.close()
        out_con_file = open(con_mat_file_path)
        self.con_mat = json.load(out_con_file)
        out_con_file.close()
        self.wih = loadtxt(wih_file_path, delimiter=',')
        self.who = loadtxt(who_file_path, delimiter=',')
        pass

    def save(self):
        dir = os.path.join('nets', 'digit_1_hidden_layer')
        FileUtils.mkdir(dir)
        params_file_path =  FileUtils.path(dir, 'params.json')
        con_mat_file_path =  FileUtils.path(dir, 'con_mat.json')
        wih_file_path =  FileUtils.path(dir, 'wih.csv')
        who_file_path =  FileUtils.path(dir, 'who.csv')

        out_param_file = open(params_file_path, "w")
        json.dump(self.params, out_param_file)
        out_param_file.close()
        out_con_file = open(con_mat_file_path, "w")
        json.dump(self.con_mat, out_con_file)
        out_con_file.close()
        savetxt(wih_file_path, self.wih, delimiter=',')
        savetxt(who_file_path, self.who, delimiter=',')
        pass

    def parameters(self) -> ({}, {}):
        return (self.params, self.con_mat)
