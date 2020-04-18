def sigmoid(x):
    return 1/(1 + pow(2.71828,-x))

def der(x):
    return x*(1-x)

#training data
training_inputs = [[0,0],[0,1],[1,0],[1,1]]
training_outputs = [0,1,1,0]

input_neurons = 2
hidden_neurons = 2
output_neurons = 1

#weight initialization
w31 = w32 = w41 = w42 = w53 = w54 = b1 = b2 = b3 = 1
w = [w31,w32,w41,w42,w53,w54]
b = [b1,b2,b3]

 

for iteration in range(20000):
    #Output from summer and activation function in hidden layer
    v = []
    y = []

    for i in range(len(training_inputs)):
        v1 = []
        y1 = []
        count1 = 0
        for j in range(hidden_neurons):
            sum = 0
            for k in range(len(training_inputs[0])):
                sum += training_inputs[i][k]*w[count1]
                count1 += 1
            sum = sum + b[j]
            v1.append(sum)
            y1.append(sigmoid(v1[j]))
        v.append(v1)
        y.append(y1)

    #Estimated Output
    op=[]
    for m in range(len(training_outputs)):
        count2 = count1
        sum = 0          
        for n in range(hidden_neurons):
            sum += y[m][n]*w[count2]
            count2 += 1
        op.append(sigmoid(sum+b[len(b)-1]))

    #calculate error
    error = []
    for i in range(len(training_outputs)):
        error.append(training_outputs[i]-op[i])

    #stopping criterion
    # sc = 0.001
    # sum = 0
    # for i in range(len(training_outputs)):
    #     sum += pow(error[i],2)
    # sum = pow(sum,0.5)
    # if(sum < sc):
    # break

    #error adjustment in output layer
    adj_op = []
    for i in range(len(training_outputs)):
        adj_op.append(der(op[i])*error[i])

    #calculate revised weights from hidden layer to output
    for i in range(hidden_neurons):
        sum = 0
        for j in range(hidden_neurons*2):
            sum += y[j][i]*adj_op[j]
        w[len(w)-hidden_neurons+i] += sum

    #adjust output bias
    sum = 0
    for i in range(len(training_outputs)):
        sum += b[2]*adj_op[i]
    b[2] += sum
    
    #adjustment in hidden layer
    adj_h1 = []     
    for i in range(hidden_neurons):
        sum1 = 0
        sum2 = 0

        for k in range(len(training_outputs)):
            sum1 += w[input_neurons*hidden_neurons+i]*adj_op[k]

        for l in range(len(training_inputs)):
            sum2 += der(y[l][i])*sum1
        adj_h1.append(sum2)

       

    #calculate revised weights from input to hidden layer     
    for k in range(input_neurons):
        for i in range(hidden_neurons):
            sum = 0
            for j in range(input_neurons*2):
                sum += training_inputs[j][i]*adj_h1[k]
            w[i] += sum

    #Calculate hidden bias
    for i in range(hidden_neurons):
        b[1] += b[1]*adj_h1[i]
        b[2] += b[2]*adj_h1[i]   