# import pandas as pd
# import numpy as np
# import math

# def load_data(filename):
#     df = pd.read_csv(filename)
#     X = df.iloc[:, :-1].values
#     y = df.iloc[:, -1].values
#     return X, y

# def separate_by_class(X, y):
#     separated = {}
#     for i in range(len(X)):
#         features = X[i]
#         class_label = y[i]
#         if class_label not in separated:
#             separated[class_label] = []
#         separated[class_label].append(features)
#     return separated

# def calculate_mean(features):
#     return sum(features) / float(len(features))

# def calculate_std_dev(features):
#     mean = calculate_mean(features)
#     variance = sum([(x - mean) ** 2 for x in features]) / float(len(features) - 1)
#     return math.sqrt(variance)

# def summarize_dataset(dataset):
#     summaries = [(calculate_mean(column), calculate_std_dev(column)) for column in zip(*dataset)]
#     return summaries

# def summarize_by_class(X, y):
#     separated = separate_by_class(X, y)
#     summaries = {}
#     for class_label, instances in separated.items():
#         summaries[class_label] = summarize_dataset(instances)
#     return summaries

# def calculate_probability(x, mean, std_dev):
#     exponent = math.exp(-((x - mean) ** 2 / (2 * std_dev ** 2)))
#     return (1 / (math.sqrt(2 * math.pi) * std_dev)) * exponent

# def calculate_class_probabilities(summaries, input_features):
#     probabilities = {}
#     for class_label, class_summaries in summaries.items():
#         probabilities[class_label] = 1
#         for i in range(len(class_summaries)):
#             mean, std_dev = class_summaries[i]
#             x = input_features[i]
#             probabilities[class_label] *= calculate_probability(x, mean, std_dev)
#     return probabilities

# def predict(summaries, input_features):
#     probabilities = calculate_class_probabilities(summaries, input_features)
#     best_class, best_prob = None, -1
#     for class_label, probability in probabilities.items():
#         if best_class is None or probability > best_prob:
#             best_class = class_label
#             best_prob = probability
#     return best_class

# def get_predictions(summaries, X_test):
#     predictions = []
#     for i in range(len(X_test)):
#         result = predict(summaries, X_test[i])
#         predictions.append(result)
#     return predictions

# def get_accuracy(predictions, y_test):
#     correct = 0
#     for i in range(len(predictions)):
#         if predictions[i] == y_test[i]:
#             correct += 1
#     return (correct / float(len(predictions))) * 100.0

# # Load the dataset
# filename = 'liris.csv'
# X, y = load_data(filename)

# # Manually split the dataset into training and test sets
# test_size = 0.25
# split_index = int(len(X) * (1 - test_size))

# X_train = X[:split_index]
# X_test = X[split_index:]
# y_train = y[:split_index]
# y_test = y[split_index:]

# # Train the Naive Bayes classifier
# summaries = summarize_by_class(X_train, y_train)

# # Make predictions on the test set
# predictions = get_predictions(summaries, X_test)


# # Calculate the accuracy of the classifier
# accuracy = get_accuracy(predictions, y_test)
# print(f"Accuracy: {accuracy:.2f}%")


from math import sqrt
from random import randrange
from random import seed
from csv import reader
from math import exp
from math import pi


def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if class_value not in separated:
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated


def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
    return sqrt(variance)


def summarize_dataset(dataset):
    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
    del (summaries[-1])
    return summaries


def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries


def calculate_probability(x, mean, stdev):
    exponent = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent


def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    return probabilities


def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label


# Naive Bayes Algorithm
def naive_bayes(train, test):
    summarize = summarize_by_class(train)
    predictions = list()
    for row in test:
        output = predict(summarize, row)
        predictions.append(output)
    return predictions


# Naive Bayes on Iris Dataset
seed(1)
dataset = load_csv('iris.csv')
for i in range(len(dataset[0]) - 1):
    str_column_to_float(dataset, i)

# Encode
str_column_to_int(dataset, len(dataset[0]) - 1)

# Evaluate
n_folds = 5
scores = evaluate_algorithm(dataset, naive_bayes, n_folds)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))

# Fit model
"""
Encoding:
[Iris-virginica] => 0
[Iris-versicolor] => 1
[Iris-setosa] => 2
"""
model = summarize_by_class(dataset)

# Define a new record
row = [5.7, 2.9, 4.2, 1.3]

# Predict the label
label = predict(model, row)
print('Data=%s, Predicted: %s' % (row, label))

