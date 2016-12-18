#!/usr/bin/env python
# -*- coding: utf-8 -*-

from LogisticRegression import LogisticRegression
from arguments_parser import parser_arguments
import matplotlib.pyplot as plt
from nltk.corpus import names
import pandas as pd
import numpy as np
import pickle
import csv
import re

"""ner.py: Name entity recognition Classifier."""
__author__ = "Anthony FARAUT"


LABELED_NAMES = ([name for name in names.words('male.txt')] + [name for name in names.words('female.txt')])
NUMBER_OF_FEATURE = 26
LEARNING_RATE = 0.0001
ITERATION = 600


def number_of_vowel(word):
    """ Get the number of vowel in the word (token)
    Parameters
    ----------
    word : The word

    Returns
    -------
    int : The number of vowel
    """
    accented_vowel_list = ['a', 'e', 'i', 'o', 'u', 'y', 'À', 'Á', 'Â', 'Ã', 'Ä', 'Å', 'È', 'É', 'Ê', 'Ë', 'Ì', 'Í',
                           'Î', 'Ï', 'Ò', 'Ó', 'Ô', 'Õ', 'Ö', 'Ù', 'Ú', 'Û', 'Ü', 'Ý', 'Ÿ', 'à', 'á', 'â', 'ã', 'ä',
                           'å', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'î', 'ï', 'ò', 'ó', 'ô', 'õ', 'ö', 'ù', 'ú', 'û', 'ü',
                           'ý', 'ÿ']
    return sum(1 for letter in word if letter in accented_vowel_list) / len(word)


def number_of_consonant(word):
    """ Get the number of consonant in the word (token)
    Parameters
    ----------
    word : The word

    Returns
    -------
    int : The number of consonant
    """
    return (len(word) - number_of_vowel(word)) / len(word)


def is_capitalized(word):
    """ Check if the word (token) is capitalized
    Parameters
    ----------
    word : The word to check

    Returns
    -------
    int : (Boolean)
    """
    if word[0].isupper():
        return 1
    return 0


def is_a_name(word):
    """ Check if the word (token) is a name accord to the nltk.corpus names
    Parameters
    ----------
    word : The word to check

    Returns
    -------
    int : (Boolean)
    """
    return 1 if word in LABELED_NAMES else 0


def is_a_number(word):
    """ Check if the word (token) is a number thanks to a regex
    Parameters
    ----------
    word : The word to check

    Returns
    -------
    int : (Boolean)
    """
    p = re.compile('^-?[0-9]+$')
    return 1 if p.match(word) else 0


def size_of_word(word):
    """ Get the length of the word (token)
    Parameters
    ----------
    word : The word

    Returns
    -------
    int : The length of the word
    """
    return len(word)


def is_all_capitalized(word):
    """ Check if all the letter from the word (token) is capitalized thanks to a regex
    Parameters
    ----------
    word : The word to check

    Returns
    -------
    int : (Boolean)
    """
    p = re.compile('^[A-Z]+$')
    return 1 if p.match(word) else 0


def is_end_of_sentence_punctuation_followed_by_capitalization_word(word_n1, word_n):
    """ Check if the word (token) is capitalized and the previous word (token) is a punctuation
    Parameters
    ----------
    word_n1 : The previous word
    word_n1 : The word to check

    Returns
    -------
    int : (Boolean)
    """
    return 1 if is_a_end_of_sentence_punctuation(word_n1) and is_capitalized(word_n) else 0


def is_a_end_of_sentence_punctuation(word):
    """ Check if the word (token) is a end of sentence punctuation
    Parameters
    ----------
    word : The word to check

    Returns
    -------
    int : (Boolean)
    """
    return 1 if word in ['.', '?', '!', '...'] else 0


def features(word_n2, word_n1, word_n, y_class):
    """ Generate the features for the tri-grams and the class
    Parameters
    ----------
    word_n2 : The word in position n-2
    word_n1 : The word in position n-1
    word_n : The word in position n
    y_class : The class for the word in position n

    Returns
    -------
    list : The list containing all the features
    """
    features_list=[]

    features_list.append(is_capitalized(word_n2))
    features_list.append(is_capitalized(word_n1))
    features_list.append(is_capitalized(word_n))

    features_list.append(is_a_name(word_n2))
    features_list.append(is_a_name(word_n1))
    features_list.append(is_a_name(word_n))

    features_list.append(is_a_number(word_n2))
    features_list.append(is_a_number(word_n1))
    features_list.append(is_a_number(word_n))

    features_list.append(size_of_word(word_n2))
    features_list.append(size_of_word(word_n1))
    features_list.append(size_of_word(word_n))

    features_list.append(number_of_vowel(word_n2))
    features_list.append(number_of_vowel(word_n1))
    features_list.append(number_of_vowel(word_n))

    features_list.append(number_of_consonant(word_n2))
    features_list.append(number_of_consonant(word_n1))
    features_list.append(number_of_consonant(word_n))

    features_list.append(is_all_capitalized(word_n2))
    features_list.append(is_all_capitalized(word_n1))
    features_list.append(is_all_capitalized(word_n))

    features_list.append(is_a_end_of_sentence_punctuation(word_n2))
    features_list.append(is_a_end_of_sentence_punctuation(word_n1))
    features_list.append(is_a_end_of_sentence_punctuation(word_n))

    features_list.append(is_end_of_sentence_punctuation_followed_by_capitalization_word(word_n2, word_n1))
    features_list.append(is_end_of_sentence_punctuation_followed_by_capitalization_word(word_n1, word_n))

    features_list.append(y_class)
    return features_list


def write_csv(df, filename):
    """ Write a DataFrame in a csv file
    Parameters
    ----------
    df : The DataFrame to write out in the file
    filename : The filename of the csv file
    """
    df.to_csv(filename, sep=';', encoding='utf-8')


def normalize_data(matrix):
    """ Normalize the data from a matrix
    Parameters
    ----------
    matrix : The matrix to normalize the data

    Returns
    -------
    list : The matrix normalized
    """
    index_to_normalize = [10, 11, 12, 13, 14, 15, 16, 17, 18]

    for index in index_to_normalize:
        matrix[str(index)] = (matrix[str(index)] - matrix[str(index)].mean()) / matrix[str(index)].std()
    return matrix


def read_data(filename):
    """ Read the data already featured from a csv file
    Parameters
    ----------
    filename : The csv file containing the data featured

    Returns
    -------
    list : The features values
    list : The classes according to the features
    """
    df = pd.read_csv(filename, sep=';', quoting=csv.QUOTE_NONE)
    df = normalize_data(df)
    possible_values = [x for x in range(1, NUMBER_OF_FEATURE + 1)]
    x_features = df.iloc[:, possible_values].values
    y_class = df.iloc[:, NUMBER_OF_FEATURE + 1].values

    return np.copy(x_features), np.copy(y_class)


def train(x_train, y_train):
    """ Train the logistic regression
    Parameters
    ----------
    x_train : The dataset x to train
    y_train : The dataset y to train (containing the classes)

    """

    print("Logistic regression ...")
    lr = LogisticRegression(n_iter=ITERATION, eta=LEARNING_RATE).fit(x_train, y_train)
    print("Saving the model ...")
    pickle.dump(lr, open("../data/train/logistic_regression_model.p", "wb"))


def predict(lr, x_test):
    """ Predict the classes thanks to the logistic regression
    Parameters
    ----------
    x_test : The dataset x to test

    Returns
    -------
    list : The classes predicted
    """
    return lr.predict(x_test)


def evaluation(prediction, y_test):
    """ Evaluate the prediction (the classes predicted) with the ground truth data (the classes)
    Parameters
    ----------
    prediction : The prediction made
    y_test : the ground truth data (the classes)
    """
    tp = sum([pred & pred for index, pred in enumerate(prediction)])
    tn = sum([1 - (pred | pred) for index, pred in enumerate(prediction)])
    fp = sum(1 for index, pred in enumerate(prediction) if pred != y_test[index] and y_test[index] == 1)
    fn = sum(1 for index, pred in enumerate(prediction) if pred != y_test[index] and y_test[index] == 0)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = (2 * precision * recall) / (precision + recall)

    print("TP ", tp)
    print("TN ", tn)
    print("FP ", fp)
    print("FN ", fn)

    print("Precision ", precision)
    print("Recall ", recall)
    print("Accuracy ", accuracy)
    print("F1 ", f1)


def plot_logistic_regression(lr):
    """ Plot the logistic regression
    Parameters
    ----------
    lr : The logistic regression object
    """
    plt.plot(range(1, len(lr.cost_) + 1), np.log10(lr.cost_))
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    title = "Logistic Regression - Learning rate", LEARNING_RATE
    plt.title(title)
    plt.tight_layout()
    plt.show()


def main():
    train_param, test_param, learning_curve_param, data_file_param = parser_arguments()

    if train_param:
        print("Read data from the file ...")
        x_train, y_train = read_data(data_file_param)
        print("Train the model ...")
        train(x_train, y_train)
    elif test_param:
        x_test, y_test = read_data(data_file_param)
        # Loading the model
        lr = pickle.load(open("../data/train/logistic_regression_model.p", "rb"))
        prediction = predict(lr, x_test)
        evaluation(prediction, y_test)
    if learning_curve_param:
        # Loading the model
        lr = pickle.load(open("../data/train/logistic_regression_model.p", "rb"))
        plot_logistic_regression(lr)


if __name__ == '__main__':
    main()
