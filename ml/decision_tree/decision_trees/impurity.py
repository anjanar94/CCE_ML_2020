# functions related to Gini Index

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

    key_func = lambda x: x[0].strip() if isinstance(x[0],str) else x[0]
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






def gini_index(scr, grp_size, total_samples):
	if total_samples == 0:
		raise 'total_samples is 0'
	return (1.0 - scr) * (grp_size / total_samples)

def score(group, classes):
    s = 0
    if len(group) == 0:
        return s
    for class_val in classes:
        prop = [row[-1] for row in group].count(class_val) / len(group)
        s += prop * prop
    return s

def calc_total_gini(groups, classes, total_samples):
    t_gini = 0.0

    for group in groups:
        scr = score(group, classes)
        t_gini += gini_index(scr, len(group), total_samples)

    return t_gini

