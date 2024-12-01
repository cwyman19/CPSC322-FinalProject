'''Charlie Wyman and Jillian Berry
CPSC322 Final Project

This file contains various utility functions
'''
import numpy as np
from tabulate import tabulate
import math

def get_frequency(list):
    '''Find frequencies of items in a list

    Args:
        list: 1 dimensional list of data (any type)
    
    Returns:
        frequency: dictionary where each item from the list is a key, with a correspnding frequency value
    '''
    frequency = {}
    
    for item in list:
        # Convert the item to a string to handle different data types consistently
        key = str(item)
        if key in frequency:
            frequency[key] += 1
        else:
            frequency[key] = 1

    return frequency

def create_data(X_fold, y_fold, X_data, y_data):
    '''Takes tuple folds and creates useable train and test datasets
    
    Args:
        X_fold (tuple): The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold
        y_fold (tuple): Parallel to X_fold
        X_data (2D list of objects): The complete dataset used to create the folds
            (X_data is necessary because folds are only indeces, not actual data)
        y_data (1D list of objects): parallel to X_data
    
    Returns:
        X_train (2D list of objects): X_train data for a classifier
        X_test (2D list of objects): X_test data for a classifier
        y_train (1D list of objects): y_train data for a classifier
        y_test (1D list of objects): y_test data for a classifier
    '''
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for i in range(len(X_fold[0])):
        X_train.append(X_data[X_fold[0][i]])
        y_train.append(y_data[y_fold[0][i]])
    for i in range(len(X_fold[1])):
        X_test.append(X_data[X_fold[1][i]])
        y_test.append(y_data[y_fold[1][i]])
    
    return X_train, X_test, y_train, y_test


def print_matrix(matrix, labels):
    '''Prints confusion matrix data with added context and cumulative totals

    Args:
        matrix (2D list of ints): confusion matrix data
        labels (1D list of objects): header data for confusion matrix
    
    '''
    column_totals = []
    for i in range(len(labels)):
        column_totals.append(0)
    labels.append("total")
    for row in matrix:
        row_total = 0
        i = 0
        for item in row:
            row_total += item
            column_totals[i] += item
            i += 1
        row.append(row_total)
    column_totals.append(sum(column_totals))
    matrix.append(column_totals)

    table_data = []
    header = [""] + labels
    table_data.append(header)

    for label, row in zip(labels, matrix):
        table_data.append([label] + row)
    print(tabulate(table_data, tablefmt="grid"))