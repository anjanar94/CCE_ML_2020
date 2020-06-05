from csv import reader
import random


def load_input_csv(filename):
    '''
    store the data from the csv file into a list of list.
    each row in the file is the list item inside list
    :param filename: Name of the csv file
    :return: list of list containing the data
    '''

    dataset = []
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for datapoint in csv_reader:
            if not datapoint:
                continue

            dataset.append(datapoint)

    return dataset


def convert_to_float(dataset, column):
    '''
    Converts the data in the given column from string to float
    :param dataset: Input data whose column's needs to be converted
    :param column: The column number
    :return: Modified dataset
    '''
    for row in dataset:
        row[column] = float(row[column].strip())
    return dataset


def split_data(dataset, test_size):
    """
    Splits the input data into training and test data based on the test size
    :param dataset: Input dataset
    :param test_size: The size of test set(in range(0-1))
    :return: training data and test data
    """

    split = int(len(dataset) * (1 - test_size))
    random.shuffle(dataset)

    train_data = dataset[:split]
    test_data = dataset[split:]
    return train_data, test_data


def get_class_lookup(dataset):
    '''
    Converts the string class name in the data set into int.
    :param dataset: Input data
    :return: Lookup table with int as key and string as value
    '''
    lookup = {}
    int_to_class_map = {}
    classes = [row[-1] for row in dataset]
    set_classes = set(classes)
    for i, value in enumerate(set_classes):
        lookup[value] = i
    for row in dataset:
        row[-1] = lookup[row[-1]]

    for key, value in lookup.items():
        int_to_class_map[value] = key
    return int_to_class_map


def euclidean_distance(data_point_1, data_point_2):
    '''
    Calculates the Euclidean distance between two data points
    :param data_point_1: A row of the dataset
    :param data_point_2: A row of the dataset
    :return: Euclidean distance
    '''
    distance = 0
    for i in range(len(data_point_1) - 1):
        distance += (data_point_1[i] - data_point_2[i]) ** 2

    return (distance) ** 0.5


def get_neighbors(data_set, test_row, k):
    '''
    Get the nearest data point for the given test data point
    :param data_set: Input data/training data
    :param test_row: Row for which the neighbors are to be found
    :param k: number of neighbors to be found
    :return: k nearest neighbors
    '''
    neighbors = []
    distances = []

    for i in range(len(data_set)):
        e_dist = euclidean_distance(data_set[i], test_row)
        distances.append((data_set[i], e_dist))
    distances.sort(key=lambda j: j[1])

    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def predict_class(data_set, test_row, k):
    '''
    Predicst the class for the given test row
    :param data_set: Input data
    :param test_row: Row for which the class is to be determined
    :param k: number of neighbors
    :return: predicted class for the test row
    '''
    neighbors = get_neighbors(data_set, test_row, k)
    classes = [row[-1] for row in neighbors]
    max_class = max(set(classes), key=classes.count)
    return max_class


def calculate_predict_accuracy(prediction_results):
    """
    Calculates the accuracy of the prediction
    :param prediction_results: The results obtained from the knn_algorithm test
    :return: accuracy
    """
    total = len(prediction_results)
    correct = 0
    for row in prediction_results:
        original_class = row[0][-1]
        predicted_class = row[1]
        if original_class == predicted_class:
            correct += 1
    accuracy = correct * 100 / total
    return accuracy


def knn_algorithm(dataset, k, test_split = 0.2):
    '''
    Implements the KNN algorithm for the given
    :param data_filename: Input data
    :param test_file: Test file containing the points for which class is to be determined
    :param k: number of neighbors
    :return: A tuple of test data, predicted class
    '''
    result = []
    #dataset = load_input_csv(data_filename)
    n_columns = len(dataset[0])
    print(f"Loaded file {data_filename} with {len(dataset)} rows and {len(dataset[0])} columns")

    for i in range(n_columns - 1):
        dataset = convert_to_float(dataset, i)

    train_data, test_data = split_data(dataset, test_split)
    lookup = get_class_lookup(train_data)

    for row in test_data:
        res = predict_class(train_data, row, k)
        result.append((row, lookup[res]))
    accuracy=calculate_predict_accuracy(result)

    return result, len(train_data), len(test_data), accuracy


def knn_predict(dataset, predict_data, k):
    '''
    Implements the KNN algorithm for the given
    :param data_filename: Input data
    :param test_file: Test file containing the points for which class is to be determined
    :param k: number of neighbors
    :return: A tuple of test data, predicted class
    '''
    result = []
    #dataset = load_input_csv(data_filename)
    n_columns = len(dataset[0])
    print(f"Loaded dataset with {len(dataset)} rows and {len(dataset[0])} columns")

    for i in range(n_columns - 1):
        dataset = convert_to_float(dataset, i)

    #test_data = load_input_csv(test_filename)
    test_data = predict_data

    for i in range(n_columns - 1):
        test_data = convert_to_float(test_data, i)

    lookup = get_class_lookup(dataset)

    for row in test_data:
        res = predict_class(dataset, row, k)
        result.append((row, lookup[res]))

    return result
