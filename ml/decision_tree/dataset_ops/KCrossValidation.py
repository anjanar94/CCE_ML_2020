import ml.decision_tree.dataset_ops.KCrossSplit as kcs
from ml.decision_tree.decision_trees.predict import predict
import ml.decision_tree.dataset_ops.SplitXy
import ml.decision_tree.dataset_ops.DatasetHandler as dh
import ml.decision_tree.decision_trees.TreeBuild as tb

import pprint

def accuarcy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def precision(tp, fp):
    return tp/(tp+fp)

def recall(tp, fn):
    return tp/(tp+fn)


def f1_score(actual, predicted):
    tp = 0.0
    fp = 0.0
    fn = 0.0
    tn = 0.0
    for i in range(len(actual)):
        if actual[i] == predicted[i] and actual[i] == '1':
            tp += 1
        if predicted[i] == '1' and actual[i] == '0':
            fp += 1
        if predicted[i] == '0' and actual[i] == '1':
            fn += 1
        if actual[i] == predicted[i] and actual[i] == '0':
            tn += 1

    prec = precision(tp, fp)
    rec  = recall (tp, fn)

    f1 = tp/(tp+((fn+fp)/2))
    return f1

def evaluate_kfold(dataset, process_dataset, max_depth, min_size, n_folds, *args):
    folds = kcs.cross_validation_split(dataset, n_folds)
    scores = list()
    best_node=[]
    training_s=[]
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])

        X_train, y_train = process_dataset(train_set)

        i = 0
        for row in X_train:
            row.append(y_train[i])
            i += 1
        node = tb.build_tree(X_train, max_depth, min_size)
        best_node.append(node)
        predicted = list()
        training = list()
        X_test, y_test = process_dataset(fold)

        for row in X_test:
            predicted.append(predict(node, row))

        for test in X_train:
            training.append(predict(node,test))

        accuracy = accuarcy_metric(actual=y_test, predicted=predicted)
        train_score = accuarcy_metric(actual=y_train, predicted=training)
        f1 = f1_score(y_test, predicted)
        scores.append({'training_score':train_score,'accuracy':accuracy, 'F1_score':f1})
        training_s.append(train_score)
    #get position for max Accuracy
    position=training_s.index(max(training_s))
    best_tree=best_node[position]



    return scores,best_tree
