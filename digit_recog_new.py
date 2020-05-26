import numpy
import matplotlib.pyplot
from skimage.color import rgb2gray
from skimage.color import rgba2rgb

class NeuralNetwork:

    def __init__(self,inp_nodes,hid_nodes,out_nodes,learn_rate):
        self.inodes = inp_nodes
        self.hnodes = hid_nodes
        self.onodes = out_nodes
        self.lr = learn_rate
        self.activation_function = lambda x: 1/(1+pow(2.718,-x))
        self.wih = numpy.random.rand(self.hnodes,self.inodes)-0.5
        self.who = numpy.random.rand(self.onodes,self.hnodes)-0.5
        pass

    def train(self,input_list,target_list):
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

    def query(self,input_list):
        inputs = numpy.array(input_list,ndmin = 2).T
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
        pass

#******************************************************************************#
#******************************************************************************#
#Training the data
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.1

n = NeuralNetwork(input_nodes, hidden_nodes,output_nodes,learning_rate)
print("Please enter the full path of the training dataset")
path = input()
training_data_file = open(path,"r")
training_data_list = training_data_file.readlines()
training_data_file.close()

epoch = 5

for e in range(epoch):
    for records in training_data_list:
        all_values = records.split(',')
        inputs = (numpy.asfarray(all_values[1:])/255 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes)+0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs,targets)

#******************************************************************************#
#******************************************************************************#
# Testing our own handwriting
print("Enter the full path of the image file")
file_path = input()
original = matplotlib.pyplot.imread(file_path)
# matplotlib.pyplot.imshow(original)
#grayscale = rgb2gray(original)
grayscale = rgb2gray(rgba2rgb(original))
img_array = grayscale.reshape(784,)
img_array = 1-img_array
# matplotlib.pyplot.imshow(img_array.reshape((28,28)),cmap = "Greys",interpolation = "None")
query = n.query(img_array)
q = numpy.argmax(query)
print(query)
print(q)

#******************************************************************************#
#******************************************************************************#
