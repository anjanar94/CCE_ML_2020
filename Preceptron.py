def sigmoid(x):
    return 1/(1 + pow(2.718,-x))

def der(x):
    return x * (1-x)

#training data
training_inputs= [[0,0,1],[1,1,1],[1,0,1],[0,1,1]]
training_outputs = [0,1,1,0]

#initialize weights
w = [1,1,1]

for iteration in range(50000):
    #result from summer
    v = []
    y = []
    for i in range(len(training_inputs)):
        sum = 0
        for j in range(len(w)):
            sum += training_inputs[i][j]*w[j]
        v.append(sum)
        y.append(sigmoid(v[i]))
    
    #calculating error
    error = []
    
    for i in range(len(training_inputs)):
        error.append(training_outputs[i]-y[i])
    
    #Error adjustment
    adj = []
    
    for i in range(len(training_inputs)):
        adj.append(error[i]*der(y[i]))
    for i in range(len(w)):
        sum = 0
        for j in range(len(training_inputs)):
            sum += training_inputs[j][i]*adj[j]
        w[i] = w[i] + sum


test_input = [1,0,0]
sum = 0
for i in range(3):
    sum += test_input[i]*w[i]

print(sigmoid(sum))
    








    