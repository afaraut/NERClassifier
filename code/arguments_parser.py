#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

"""arguments_parser.py: Parse the arguments from the command line."""
__author__ = "Anthony FARAUT"


def parser_arguments():
    """ Parse the arguments from the command line arguments

    Returns
    -------
    bool : Train parameter
    bool : Test parameter
    bool : LEarning curve parameter
    file : The file (training or test data)
    """
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-test",
                        help="test the classifier based on the data file, outputting the f1-score, accuracy and the confusion matrix.",
                        action="store_true")
    group.add_argument("-train", help="train the classifier based on the data file.", action="store_true")
    parser.add_argument("-learning_curve", help="plots the learning_curve.", action="store_true")
    parser.add_argument('data_file', type=argparse.FileType('r'), help="training or test data.")
    args = parser.parse_args()
    return args.train, args.test, args.learning_curve, args.data_file
