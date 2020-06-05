import sys
sys.path.insert(0, './Dataset-Ops/')
sys.path.insert(1, './Decision-Trees')

import DatasetHandler as dh
import KCrossValidation as kcv
import SplitXy
import warnings

def process_dataset_titanic(train_set):
    X_train, y_train = SplitXy.splitXy(train_set)

    # Let's just drop the `Name` column and make a rough prediction
    # The name column is at index `1`
    X_train = dh.drop(X_train, column=1)
    X_train = dh.make_unique_columns_with(X_train, column=1)
    for i in range(5):
        X_train = dh.str_col_float(X_train, i)
    return X_train, y_train

def process_dataset_banknote(dataset):
    X_train, y_train = SplitXy.splitXy(dataset)
    return X_train, y_train

#filename = './dataset/titanic_1.csv'
filename = './dataset/banknote.csv'

dataset = dh.read_csv(filename, headers=False)





def calculate(acc_f1_list):
    train_op=0
    score=0
    f1_score=0
    for data in acc_f1_list:
        score+=data['accuracy']
        f1_score+=data['F1_score']
        train_op += data['training_score']

    avg_score=score/len(acc_f1_list)
    avg_f1_score=f1_score/len(acc_f1_list)
    avg_train_score = train_op / len(acc_f1_list)
    return avg_score,avg_f1_score,avg_train_score



def call1():
    acc_f1_list,best_tree =  kcv.evaluate_kfold(dataset, process_dataset_banknote, max_depth=max_depth, min_size=min_size, n_folds=folds)
    avg_score,avg_f1_score,avg_train_score=calculate(acc_f1_list)
    print(best_tree)
    print('Prediced Score  :  ',avg_score)
    print('F1 Score : ', avg_f1_score)

def call2():
    import pandas as pd
    import matplotlib.pyplot as plt
    data=[]
    for depth in max_depth:
        acc_f1_list,best_tree =  kcv.evaluate_kfold(dataset, process_dataset_banknote, max_depth=depth, min_size=min_size, n_folds=folds)
        avg_score,avg_f1_score,avg_train_score = calculate(acc_f1_list)
        pass_val=[depth,avg_train_score,avg_score,avg_f1_score]
        data.append(pass_val)
        #print(best_tree)
    df=pd.DataFrame(data,columns=['max_depth','avg_train_score','avg_test_score','avg_f1_score'])
    print(df)
    plt.figure(figsize=(12,6))
    plt.plot(df['max_depth'],df['avg_train_score'],marker='o')
    plt.plot(df['max_depth'],df['avg_test_score'],marker='o')
    plt.xlabel('Depth of Tree')
    plt.ylabel('Performance')
    plt.legend()
    plt.show()



# First Argument is max_depth
# Second Argument is min_size
max_depth = 0
min_size  = 0
folds = 5

try:
    max_depth = int(sys.argv[1])
    min_size = int(sys.argv[2])

    print ('Using:\nmax_depth =', max_depth, '\nmin_size =', min_size, '\n')
    call1()

except IndexError:
    warnings.warn('NOTE: One of the arguments is missing\nUsing Default parameters ' \
                  'for both max_depth and min_size\nSetting:\nmax_depth = 2 to 15 \nmin_size = 10 \n', stacklevel=2)
    max_depth = range(2,15)
    min_size = 10
    folds=5
    call2()
