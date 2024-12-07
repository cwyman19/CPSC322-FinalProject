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

def compute_categorical_distance(v1, v2):
    '''Compute the distance between discrete values in two parallel lists

    Args:
        v1: (1D list of objects)
        v2: (1D list of objects)

    Returns:
        distance (float)
    '''
    count = 0
    for i in range(len(v1)):
        if v1[i] == v2[i]:
            count += 1
    distance = count / len(v1)
    return distance

def tdidt(current_instances, available_attributes, att_header, attribute_domain, 
          y_labels, used_attributes, used_instances):
    '''Recursive function that runs the tdidt algorithm to build a decision tree

    Args:
        current_instances (2D list of obj): Current dataset (X_train and y_train stitched together)
        available_attributes (1D list of obj): list of attributes that have yet to be split on
        att_header (1D list of obj): Every attribute in the dataset
        attribute_domain (dict): A dictionary of all corresponding values for each attribute
        y_labels (1D list of obj): A list of each unique possible target y value
        used_attributes (1D list of obj): All previously used attributes from the previous tdidt calls
        used_instances (nD list of obj): All previous versions of the "current instances" dataset
    
    Returns:
        tree (nD list of obj): Decision tree built from current instances dataset
    '''

    split_attribute = select_attribute(current_instances, available_attributes, attribute_domain, y_labels)
    #print("splitting on:", split_attribute)
    available_attributes.remove(split_attribute) # can't split on this attribute again in this subtree
    
    used_attributes.append(split_attribute)
    used_instances.append(current_instances) # keeping a record of these for Case 3

    tree = ["Attribute", split_attribute]
    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(current_instances, split_attribute, att_header, attribute_domain)
    #print("partitions:", partitions)
    # for each partition, repeat unless one of the following occurs (base case)
    for att_value in sorted(partitions.keys()): # process in alphabetical order
        att_partition = partitions[att_value]
        value_subtree = ["Value", att_value]
        #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(att_partition) > 0 and all_same_class(att_partition):
            #print("CASE 1")
            leaf_node = ["Leaf", att_partition[0][-1], len(att_partition), len(current_instances)]
            value_subtree.append(leaf_node)
            #print("value_subtree: ", value_subtree)

        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        #   If there is an equal number, choose the first one alphabetically
        elif len(att_partition) > 0 and len(available_attributes) == 0:
            #print("CASE 2")
            votes = get_majority_vote(att_partition, y_labels) # return votes in a dictionary
            max_value = max(votes.values())
            keys_with_max_value = [key for key, value in votes.items() if value == max_value]
            if len(keys_with_max_value) >= 0:
                keys_with_max_value = sorted(keys_with_max_value)
            sorted_keys = []
            keys = sorted(votes.keys())
            # this logic will allow for votes to have a tie for max value as well as other lesser values (if labels > 2)
            for key in keys_with_max_value:
                sorted_keys.append(key)
            for key in keys:
                if key not in sorted_keys: 
                    sorted_keys.append(key)
            
            # creating majority vote leaf nodes:
            leaf_node = ["Leaf", sorted_keys[0], votes[sorted_keys[0]], sum(votes.values())]
            value_subtree.append(leaf_node)
            #print("value_subtree: ", value_subtree)



        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        #       Backtrack does NOT MEAN backtrack recursively, it just means change your mind about splitting on this attribute
            # if you get a case three, splitting on this attribute is a bad idea because at least one partition is empty
        elif len(att_partition) == 0:
            #print("CASE 3")
            #print(tree)

            leaf_total = len(used_instances[-2])
            partition_total = 0
            partitions = partition_instances(current_instances, split_attribute, att_header, attribute_domain)
            #print("partitions: ", partitions)

            for att_value in sorted(partitions.keys()): # process in alphabetical order
                att_partition = partitions[att_value]
                if sorted(partitions.keys()).index(att_value) == 0:
                    leaf_partition = att_partition[0][-1]
                if len(att_partition) > 0:
                    value_subtree = ["Value", att_value]
                    #leaf_node = ["Leaf", att_partition[0][-1], len(att_partition), len(current_instances)]
                    partition_total += len(att_partition)
            
            leaf_node = ["Leaf", leaf_partition, partition_total, leaf_total]
            tree = leaf_node
            #print("tree again: ", tree)
            return tree
        
        else:
            # none of base cases were true, recurse!!
            subtree = tdidt(att_partition, available_attributes.copy(), att_header, attribute_domain, y_labels, used_attributes, used_instances)
            value_subtree.append(subtree)

        tree.append(value_subtree)

    return tree


def select_attribute(instances, attributes, att_domain, y_labels):
    '''Selects the attribute with the lowest entropy value
    
    Args:
        instances (2D list of obj): Current dataset (X_train and y_train stitched together)
        attributes (1D list of obj): list of attributes that have yet to be split on
        att_domain (dict): A dictionary of all corresponding values for each attribute
        y_labels (1D list of obj): A list of each unique possible target y value
    
    Returns:
        attributes[att_entropies.index(min(att_entropies)) (str):
            This is the attribute with the lowest entropy value
    '''
    att_entropies = [] #this is parallel to the available_attributes list

    for att in attributes:
        att_entropy = []
        value_totals = []
        for i in range(len(att_domain[att])):
            att_entropy.append(0) # the weighted average of this list will calculate the entropy for this attribute
            value_totals.append(0)
        att_total = 0
        att_index = int(att[-1])
        for value in att_domain[att]:
            value_entropy = []
            for i in range(len(y_labels)):
                value_entropy.append(0)
            value_total = 0
            for instance in instances:
                if instance[att_index] == value:
                    att_total += 1
                    value_total += 1
                    for label_index in range(len(y_labels)):
                        if y_labels[label_index] == instance[-1]:
                            value_entropy[label_index] += 1
            #print("value entropy list: ", value_entropy)
            #print("value total: ", value_total)
            # calculating partition entropy
            for value_index in range(len(value_entropy)):
                if value_index == 0:
                    if (value_entropy[value_index] == 0): #this will prevent log2(0) from throwing an error
                        partition = 0
                    else:
                        partition = -((value_entropy[value_index]/value_total) * math.log2(value_entropy[value_index]/value_total))
                else:
                    if (value_entropy[value_index] == 0):
                        partition += 0
                    else:
                        partition -= ((value_entropy[value_index]/value_total) * math.log2(value_entropy[value_index]/value_total))
            #print("partition entropy: ", partition)
            att_entropy.append(partition)
            value_totals.append(value_total)
        
        # calculating attribute entropy
        entropy = 0
        for i in range(len(att_entropy)):
            entropy += (value_totals[i]/att_total * att_entropy[i])
        att_entropies.append(entropy)
    
    #print("attribute entropies: ", att_entropies)
    return attributes[att_entropies.index(min(att_entropies))]


def partition_instances(instances, attribute, att_header, attribute_domains):
    '''Groupby function that groups current_instances by the chosen split_attribute

    Args:
        instances (2D list of obj): Current dataset (X_train and y_train stitched together)
        attribute (str): Attribute that has been chosen to split on
        att_header (1D list of obj): Every attribute in the dataset
        attribute_domains (dict): A dictionary of all corresponding values for each attribute
    
    Returns:
        partitions (dict): A dictionary with instances data grouped by attribute
    '''

    # this is group by attribute domain (not values of attribute in instances)
    att_index = att_header.index(attribute)
    att_domain = attribute_domains[attribute]
    partitions = {}
    for att_value in att_domain: # "Junior" -> "Mid" -> "Senior"
        partitions[att_value] = []
        for instance in instances:
            if instance[att_index] == att_value:
                partitions[att_value].append(instance)

    return partitions


def all_same_class(instances):
    '''Check if instances is made up of all one class (for finding leaf nodes)
    
    Args:
        instances (list of obj): one partition from the partitioned dataset

    Returns:
        bool
    '''
    first_class = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_class:
            return False
    # get here, then all same class labels
    return True 

def get_majority_vote(partition, labels):
    '''Find majority votes for making a majority vote leaf node

    Args:
        partition (list of obj): data used for finding majority
        labels (list of obj): list of possible classes
    
    Returns:
        majority_votes (dict): Dictionary showing vote totals for  labels

    Note: This does not return the majority. It returns all voting data, from which majority
        can be found later.
    '''
    majority_votes = {}
    for label in labels:
        majority_votes[label] = 0
        for instance in partition:
            if instance[-1] == label:
                majority_votes[label] += 1
    
    return majority_votes

def tdidt_predict(tree, instance, header):
    '''Predict an unseen instance with a decision tree model

    Args:
        tree(nested (list): Decision tree dataset
        instance(list of obj): One instance from an X_test dataset
        header(list of obj): list of all attributes
    
        Returns:
            tree[1] (list of obj): Conditional return that triggers if the function is at a leaf node
            tdidt_predict: Recursive call if the function is not at a leaf node
    '''
    # base case: we are at a leaf node and can return the class prediction
    info_type = tree[0] # "Leaf" or "Attribute"
    if info_type == "Leaf":
        return tree[1] # class label
    
    # if we are here, we are at an Attribute
    # we need to match the instance's value for this attribute
    # to the apprppriate subtree
    att_index = header.index(tree[1])
    for i in range(2, len(tree)):
        value_list = tree[i]
        # do we have a match with instance for this attribute?
        if value_list[1] == instance[att_index]:
            return tdidt_predict(value_list[2], instance, header) # if we have a match, pass in the subtree

def generate_combinations(values, index=0, current_combination=[]):
    '''Finds every possible combination of attribute values
    
    Args:
        values (list of obj): every possible value of the dataset
        index (int): iterator for keeping track of values
        current_combination (list of obj): combination of values used to build the return statement

    Returns:
        Combinations (2D list of obj): Every possible combination of attribute values
    '''
    # if we've processed all lists, return the current combination
    if index == len(values):
        return [current_combination]

    # process each element in the current list (values[index])
    combinations = []
    for value in values[index]:
    
        new_combination = current_combination + [value]
        combinations.extend(generate_combinations(values, index + 1, new_combination))

    return combinations