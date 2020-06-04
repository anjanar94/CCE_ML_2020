import os
import string
import sys
import math
import random
import os
from collections import Counter
import itertools
from math import log
import time

# Implement your decision tree below
# Used the ID3 algorithm to implement the Decision Tree

# Class used for learning and building the Decision Tree using the given Training Set
class DecisionTree:
    tree = {}
    def learn(self, training_set, attributes, target):
        self.training_set=training_set
        self.attributes=attributes
        self.target=target
        self.tree = build_tree(training_set, attributes, target)
        return self.tree
    def predict(self,test_set):
        #print (self.tree)
        predicted_label=[]
        #print(test_set)
        training_outputs=[val[-1] for val in self.training_set]
        #print(training_outputs)
        for row in test_set:
            if len(row)==len(self.training_set[0]):
                X_test=row[:-1]
            else:
                X_test=row
            predicted_value=classify_prediction(self.tree,X_test,self.attributes,training_outputs)
            predicted_label.append(predicted_value)
        return predicted_label
    def score(self,x_test,y_predict):
        x_output=[val[-1] for val in x_test]
        total_record=len(x_output)
        match_records_position=[i for i, j in zip(x_output, y_predict) if i == j]
        score=(len(match_records_position)/total_record)
        return score

def classify_prediction(model,example,test_col_name,training_outputs):
    #print(example)
    question=list(model.keys())[0]
    vals=model[question]
    question_position=test_col_name.index(question)
    example_data=example[question_position]
    try:
        model=vals[example_data]
    except:
        #Code required modification for getting correct predicted value
        model=random.choice(training_outputs)
    if not isinstance(model, dict):
        return model
    else:
        return classify_prediction(model,example, test_col_name,training_outputs)

#Step 1 - function call required
def read_files( filepath):
    '''
    input - File path of input file.
    read file and convert into 2d array.
    handles only csv format.
    returns Column header and dataset into two different variable.

    '''
    datastore=[]
    final_dataset ={}
    with open(filepath,'r') as files:
        for line in files.readlines():
            lines=line.strip().split(',')
            for i in range(len(lines)):
                if len(lines[i])==0 or len(lines[i])== None:
                    lines[i]=None
                elif lines[i][0]=='"' or lines[i][-1]=='"':
                    lines[i]=lines[i][1:-1]
                    value=digit_check(lines[i])
                else:
                    value=digit_check(lines[i])
                if value=='int':
                    lines[i]=int(lines[i])
                elif value=='float':
                    lines[i]=float(lines[i])
                else:
                    lines[i]=lines[i]
            datastore.append(lines)

    columns=datastore[0]
    dataset=datastore[1:]

    return columns,dataset


#Step 1.1
def digit_check(user_input):
    '''
    convert string digits into integer / float.
    inputs each element of 2d array.
    return int/float/string to read_files function.
    required - 2d array reads all elements as string char.
    '''
    try:
       val = int(user_input)
       return 'int'
    except ValueError:
      try:
        val = float(user_input)
        return 'float'
      except ValueError:
          return 'string'



#Step 3.1

def random_number(low, high):
    """
    a time based random number generator
    uses the random time between a user's input events
    returns an integer between low and high-1
    """
    return int(low + int(time.time()*1000) % (high - low))



#Step 3.2
def random_indices(high,test_size):
    '''
    generates random sample index.
    inputs length of dataset and sample size
    returns radom indices

    '''
    test_indices=[]
    while len(test_indices)<test_size:
        indices=random_number(len(test_indices),high)
        test_indices.append(indices)
        test_indices = list( dict.fromkeys(test_indices) )
    test_indices.sort()
    return test_indices




# Step 3 - function call required
def test_train_split(df,test_size):
    '''
    input dataset and test size
    returns test and training data set

    '''
    high=len(df)
    train_df=[]
    if isinstance(test_size,float):
        test_size=round(test_size*len(df))
    test_indices = random_indices(high,test_size)
    test_df=[df[value] for value in test_indices]
    for i in range(len(df)):
        if i not in test_indices:
            train_df.append(df[i])
    return test_df,train_df

# Majority Function which tells which class has more entries in given data-set
def default_Y(attributes, data, target):

    freq = {}
    index = attributes.index(target)

    for tuple in data:
        if (tuple[index] in freq.keys()):
            freq[tuple[index]] += 1
        else:
            freq[tuple[index]] = 1

    max = 0
    major = ""

    for key in freq.keys():
        if freq[key]>max:
            max = freq[key]
            major = key

    return major


# This function will get unique values for that particular attribute from the given data
def get_values(data, attributes, attr):

    index = attributes.index(attr)
    values = []

    for entry in data:
        if entry[index] not in values:
            values.append(entry[index])

    return values

# This function will get all the rows of the data where the chosen "best" attribute has a value "val"
def get_data(data, attributes, best, val):

    new_data = [[]]
    index = attributes.index(best)

    for entry in data:
        if (entry[index] == val):
            newEntry = []
            for i in range(0,len(entry)):
                if(i != index):
                    newEntry.append(entry[i])
            new_data.append(newEntry)

    new_data.remove([])
    return new_data


# This function is used to build the decision tree using the given data, attributes and the target attributes.
# It returns the decision tree in the end.
def build_tree(data, attributes, target):

    data = data[:]
    #print(data)
    vals = [record[attributes.index(target)] for record in data]
    #print("vals: ", vals)
    default = default_Y(attributes, data, target)
    #print("default after calling default_Y: ", default)
    if not data or (len(attributes) - 1) <= 0:
        return default
    elif vals.count(vals[0]) == len(vals):
        return vals[0]
    else:
        #best = attr_choose(data, attributes, target)
        best = attr_choose_new(data, attributes, target)
        #print("best column to split",best)
        tree = {best:{}}

        for val in get_values(data, attributes, best):
            new_data = get_data(data, attributes, best, val)
            newAttr = attributes[:]
            newAttr.remove(best)
            subtree = build_tree(new_data, newAttr, target)
            tree[best][val] = subtree

    return tree

def entropy(pi):
    '''
    return the Entropy of a probability distribution:
    entropy(p) = − SUM (Pi * log(Pi) )
    defintion:
            entropy is a metric to measure the uncertainty of a probability distribution.
    entropy ranges between 0 to 1
    Low entropy means the distribution varies (peaks and valleys).
    High entropy means the distribution is uniform.

    '''

    total = 0
    for p in pi:
        p = p / sum(pi)
        if p != 0:
            total += p * log(p, 2)
        else:
            total += 0
    total *= -1
    return total


def gain(Y_count_list, feature_list):
    '''
    return the information gain:
    gain(D, A) = entropy(D)−􏰋 SUM ( |Di| / |D| * entropy(Di) )
    '''

    total = 0
    for v in feature_list:
        total += sum(v) / sum(Y_count_list) * entropy(v)

    gain = entropy(Y_count_list) - total
    return gain

def convert_feature_to_x_y_relation( feature):
    '''
    given a feature, this will give all [x,y] in groups
    '''
    feature_tuple = (tuple(f) for f in feature)
    #print(feature_tuple)


    ls = []
    c= Counter(feature_tuple)
    #print("Counter: ",c)
    for l in c:
        ls.append([l[0] , l[1], c[l]])
    #print("ls",ls)

    key_func = lambda x: x[0].strip()
    ls.sort()
    gr = []

    for key, group in itertools.groupby(ls, key_func):
       #print("key_func",key, list(group))
       gr.append(list(group))
     #print("gr", gr)

    gr.sort()
    final_list = []
    for i,grs in enumerate(gr):
        if len(grs) == 2 :
            final_list.append([gr[i][0][2],gr[i][1][2]])
        elif gr[i][0][1] == 0:
            final_list.append([gr[i][0][2],0])
        elif gr[i][0][1] == 1:
            final_list.append([0,gr[i][0][2]])
    return final_list

#############################################Variable dist is a dictionary with Key as feature and value as a list giving feature value and dependent variable#######################################################
def get_dictionary_with_x_groups(columname,X,Y):
    Variable_dict ={}
    for colm in  range(len(columname)-1):
        temp=[]
        for x,y in zip([x[colm] for x in X] ,Y):
            temp.append([x,y])
        Variable_dict[columname[colm]] = temp
        temp.append([x,y])
    return Variable_dict

def give_x_information_gain(Variable_dict, Y_count_list, attributes):
    root={}
    groups={}
    for key in Variable_dict.keys():
        #print("Groups for Feature %s is %s" %(key,  dt.convert_feature_to_x_y_relation(Variable_dict[key])))
        #print("Information Gain for Feature %s is %s" %(key,  dt.gain(Y_count_list,dt.convert_feature_to_x_y_relation(Variable_dict[key]))))
        root[key] =  gain(Y_count_list,convert_feature_to_x_y_relation(Variable_dict[key]))
        groups[key] = Variable_dict[key]
    Max_IG = max(root, key=root.get)
    root_index = attributes.index(Max_IG)
    Max_IG_Value = root[Max_IG]
    return Max_IG,root_index,Max_IG_Value

def attr_choose_new(data, attributes, target):
    Variable_dict ={}
    #Below gives dataset as dictionary of all columns as key and value as a list of [value of x, value of Y]
    X = [x[:-1] for x in data]
    Y = [x[-1] for x in data]
    #print("X",X)
    #print("Y",Y)
    Y_count_list = [Y.count(y) for y in set(Y)]
    Variable_dict = get_dictionary_with_x_groups(attributes,X,Y)
    #print("All rows as value and key as column", Variable_dict)
    #print("\n")
    Max_IG,root_index,Max_IG_Value = give_x_information_gain(Variable_dict , Y_count_list, attributes)
    return attributes[root_index]
