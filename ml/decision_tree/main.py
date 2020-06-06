import sys
sys.path.insert(0, './Dataset-Ops/')
sys.path.insert(1, './Decision-Trees')

import ml.decision_tree.dataset_ops.DatasetHandler as dh
import ml.decision_tree.dataset_ops.KCrossValidation as kcv
import ml.decision_tree.dataset_ops.SplitXy as spxy
import warnings

def train(file, maxdepth, minsize, folds):
    print('Decision Tree New API!!')
    dataset = dh.read_csv(file, headers=False)
    acc_f1_list, best_tree =  kcv.evaluate_kfold(dataset, process_dataset_banknote, max_depth=maxdepth, min_size=minsize, n_folds=folds)
    avg_score,avg_f1_score,avg_train_score = calculate(acc_f1_list)
    print(best_tree)
    print('Prediced Score  :  ',avg_score)
    print('F1 Score : ', avg_f1_score)
    return best_tree, avg_score, avg_f1_score

def process_dataset_banknote(dataset):
    X_train, y_train = spxy.splitXy(dataset)
    return X_train, y_train


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
