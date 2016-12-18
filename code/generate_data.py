#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ner import features, write_csv
import pandas as pd
import numpy as np
import csv

"""generate_data.py: Generate and store the data features."""
__author__ = "Anthony FARAUT"

NUMBER_OF_FEATURE = 26


def generate_data(filename_in, filename_out):
    """ Generate and store the data features
    Parameters
    ----------
    filename_in : The name of the csv file containing the data
    filename_out : The name of the csv file in which it will be saved the data featured
    """
    file_in = open(filename_in, 'r')
    file_out = open(filename_out, 'w+')

    df = pd.read_csv(file_in, header=None, sep=' ', quoting=csv.QUOTE_NONE)
    x = df.iloc[:, 0].values
    y_class = df.iloc[:, -1].values
    file_in.close()

    y_class = np.where(y_class == 'O', 0, 1)

    x_features = []
    size_x = len(x)
    for i in range(3, size_x):
        if i % 5000 == 0:
            print(i, "/", size_x)
        x_features.append(features(x[i-2], x[i-1], x[i], y_class[i]))

    df_write = pd.DataFrame(x_features)

    tab = [x for x in range(1, NUMBER_OF_FEATURE + 2)]
    df_write.columns = tab
    write_csv(df_write, file_out)
    file_out.close()


def main():
    generate_data('../data/in/ned.train', '../data/out/ned_train.csv')
    print("ned train generated ... 1/6")
    generate_data('../data/in/ned.testa', '../data/out/ned_testa.csv')
    print("ned testa generated ... 2/6")
    generate_data('../data/in/ned.testb', '../data/out/ned_testb.csv')
    print("ned testb generated ... 3/6")

    generate_data('../data/in/esp.train', '../data/out/esp_train.csv')
    print("esp train generated ... 5/6")
    generate_data('../data/in/esp.testa', '../data/out/esp_testa.csv')
    print("esp testa generated ... 5/6")
    generate_data('../data/in/esp.testb', '../data/out/esp_testb.csv')
    print("esp testb generated ... 6/6")


if __name__ == '__main__':
    main()
