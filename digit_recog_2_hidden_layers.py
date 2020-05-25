import numpy
import matplotlib.pyplot

class neuralNetwork:
    def __init__(self,inputnodes,hiddennodes1,hiddennodes2,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes1 = hiddennodes1
        self.hnodes2 = hiddennodes2
        self.onodes = outputnodes
        self.lr = learningrate
        self.activation_function = lambda x: 1/(1+pow(2.718,-x))
        self.wih = numpy.random.rand(self.hnodes1,self.inodes)-0.5
        self.whh2 = numpy.random.rand(self.hnodes2,self.hnodes1)-0.5
        self.who = numpy.random.rand(self.onodes,self.hnodes2)-0.5
        pass

    def train(self,input_list,target_list):
        inputs = numpy.array(input_list,ndmin = 2).T
        targets = numpy.array(target_list, ndmin = 2).T
        hidden_inputs1 = numpy.dot(self.wih,inputs)
        hidden_outputs1 = self.activation_function(hidden_inputs1)
        hidden_inputs2 = numpy.dot(self.whh2,hidden_outputs1)
        hidden_outputs2 = self.activation_function(hidden_inputs2)       
        final_inputs = numpy.dot(self.who,hidden_outputs2)
        final_outputs = self.activation_function(final_inputs)
        output_errors = targets - final_outputs
        hidden_errors2 = numpy.dot(self.who.T,output_errors)
        hidden_errors1 = numpy.dot(self.whh2.T,hidden_errors2)
        self.who += self.lr * numpy.dot((output_errors*final_outputs*(1-final_outputs)),numpy.transpose(hidden_outputs2))
        self.whh2 += self.lr * numpy.dot((hidden_errors2*hidden_outputs2*(1-hidden_outputs2)),numpy.transpose(hidden_outputs1))
        self.wih += self.lr * numpy.dot((hidden_errors1*hidden_outputs1*(1-hidden_outputs1)),numpy.transpose(inputs))
        pass

    def query(self,input_list):
        inputs = numpy.array(input_list,ndmin = 2).T
        hidden_inputs1 = numpy.dot(self.wih,inputs)
        hidden_outputs1 = self.activation_function(hidden_inputs1)
        hidden_inputs2 = numpy.dot(self.whh2,hidden_outputs1)
        hidden_outputs2 = self.activation_function(hidden_inputs2)       
        final_inputs = numpy.dot(self.who,hidden_outputs2)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
        pass

#Training the data   
input_nodes = 784
hidden_nodes1 = 100
hidden_nodes2 = 50
output_nodes = 10
learning_rate = 0.1

n = neuralNetwork(input_nodes, hidden_nodes1,hidden_nodes2,output_nodes,learning_rate)

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

#Testing the data
test_data_file = open("C:/Users/Dell/Desktop/Machine Learning/project/mnist_dataset/mnist_test.csv","r")
test_data_list = test_data_file.readlines()
test_data_file.close()

scorecard = []

for record in test_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:])/255*0.99)+0.01
    correct_label = int(all_values[0])
#    print(correct_label,"correct label")
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
#    print(label,"network's answer")
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)

scorecard_array = numpy.asarray(scorecard)
print("performance = ",scorecard_array.sum()/scorecard_array.size*100,"%",sep='')

##Displaying one single test data    
#all_val = test_data_list[2092].split(',')
#image_arr = (numpy.asfarray(all_val[1:])).reshape((28,28))
#matplotlib.pyplot.imshow(image_arr,cmap='Greys',interpolation = 'None')
#input = (numpy.asfarray(all_val[1:])/255*0.99)+0.01
#numpy.argmax(n.query(input))

##Predicting User Input
#from skimage.color import rgb2gray
#original = matplotlib.pyplot.imread("C:/Users/Dell/Desktop/Machine Learning/project/mnist_dataset/Images/four.png")
#grayscale = rgb2gray(original)
#img_array = grayscale.reshape(784,)
#img_array = 1-img_array
#matplotlib.pyplot.imshow(img_array.reshape((28,28)),cmap = "Greys",interpolation = "None")

#numpy.argmax(n.query(img_array))