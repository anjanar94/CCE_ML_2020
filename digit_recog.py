import numpy
import matplotlib.pyplot

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
training_data_file = open("C:/Users/Dell/Desktop/Machine Learning/project/mnist_dataset/mnist_train.csv","r")
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
#Testing the data
test_data_file = open("C:/Users/Dell/Desktop/Machine Learning/project/mnist_dataset/mnist_test.csv","r")
test_data_list = test_data_file.readlines()
test_data_file.close()

#Confusion Matrix
conf_mat = []
for i in range(11):
    l = []
    for j in range(11):
        l.append(0)
    conf_mat.append(l)

for i in range(11):
    if(i == 0):
        conf_mat[i][i] = 0
    else:
        conf_mat[0][i] = i-1
        conf_mat[i][0] = i-1
        
result = []

for record in test_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:])/255*0.99)+0.01
    correct_label = int(all_values[0])
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    if (label == correct_label):
        result.append(1)
    else:
        result.append(0)
    conf_mat[label+1][correct_label+1] += 1

result_array = numpy.asarray(result)
print("performance = ",result_array.sum()/result_array.size*100,"%",sep='')

#******************************************************************************#
#******************************************************************************#
#Output from a single input from test data
all_val = test_data_list[16].split(',')
image_arr = (numpy.asfarray(all_val[1:])).reshape((28,28))
matplotlib.pyplot.imshow(image_arr,cmap='Greys',interpolation = 'None')

input = (numpy.asfarray(all_val[1:])/255*0.99)+0.01
numpy.argmax(n.query(input))

#******************************************************************************#
#******************************************************************************#
# Testing our own handwriting
from skimage.color import rgb2gray

print("Enter the full path of the file")
file_path = input()
original = matplotlib.pyplot.imread(file_path)
# matplotlib.pyplot.imshow(original)
grayscale = rgb2gray(original)
img_array = grayscale.reshape(784,)
img_array = 1-img_array
matplotlib.pyplot.imshow(img_array.reshape((28,28)),cmap = "Greys",interpolation = "None")

numpy.argmax(n.query(img_array))

#******************************************************************************#
#******************************************************************************#
                







