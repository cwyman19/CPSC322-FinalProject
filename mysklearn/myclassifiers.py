import math
import numpy as np
from mysklearn import myutils
import operator
from mysklearn import myevaluation

'''Charlie Wyman and Jillian Berry
CPSC322: Final Project

This file contains five different classifiers:
    MyKNeighborsClassifier
    MyDummyClassifier
    MyNaiveBayesClassifier
    MyDecisionTreeClassifier
    MyRandomForestClassifier
'''

def compute_euclidean_distance(v1, v2):
    """returns all the euclidiean distance

        Args:
            v1(list of numeric vals): stores a list of numerical values
            v2 (list of numeric values): stores the list of numerical values

        Returns:
           distance (int): returns the total distance
    """
    distance = 0
    size = len(v1)
    for count in range(size):
        distance = distance + ((v1[count] - v2[count]) ** 2)
    total = (distance) ** (0.5)
    return total

def compute_distance(variable_1, variable_2):
    """Computes the distance between two instances, supporting both numeric and categorical features.

    Args:
        variable_1 (list): A list representing the first instance (sample).
        variable_2 (list): A list representing the second instance (sample).

    Returns:
        float: The Euclidean distance between the two instances.
    """
    distance = 0
    if isinstance(variable_1[0], str):
        for val1, val2 in zip(variable_1, variable_2):
            distance += 0 if val1 == val2 else 1
        distance = distance ** 0.5
    else:
        distance = compute_euclidean_distance(variable_1, variable_2)
    return distance # Return the Euclidean distance

def sort(sub_list):
    """sorts the sub_list from smallest to largest

        Args:
            sublist (2d list): a sub_list of numerical values

    """
    size = len(sub_list)
    for index in range(0, size):
        for next_index in range(0, size-index-1):
            if sub_list[next_index][1] > sub_list[next_index + 1][1]:
                tempo = sub_list[next_index]
                sub_list[next_index] = sub_list[next_index + 1]
                sub_list[next_index] = sub_list[next_index + 1]
                sub_list[next_index + 1] = tempo

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
        self.type = None

    def fit(self, X_train, y_train, type="numeric"):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train
        self.type = type


    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """

        neighbor_indeces = []
        distances = []

        #this is in a loop incase X_test has multiple testing samples
        for instance in (X_test):
            temp_row_indeces = []
            if self.type == "numeric":
                for i, row in enumerate(self.X_train):
                    dist = compute_euclidean_distance(row, instance)
                    temp_row_indeces.append((i, dist))
            else:
                for i, row in enumerate(self.X_train):
                    dist = compute_distance(row, instance)
                    temp_row_indeces.append((i, dist))
            
            # need to sort row_indexes_dists by dist
            temp_row_indeces.sort(key=operator.itemgetter(-1)) # get the item
            # in the tuple at index -1 and use that for sorting
            k = self.n_neighbors
            top_k = temp_row_indeces[:k]
            row_indeces = []
            row_dists = []
            for row in top_k:
                row_indeces.append(row[0])
                row_dists.append(row[1])
            neighbor_indeces.append(row_indeces)
            distances.append(row_dists)


        return distances, neighbor_indeces

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """

        y_nearest = []
        y_pred = []
        distances, indexes = self.kneighbors(X_test)

        for row in indexes:
            for item in row:
                y_nearest.append(self.y_train[item])
            frequency_count = myutils.get_frequency(y_nearest)
            y_pred.append(max(frequency_count, key=frequency_count.get))

        return y_pred

'''
class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closest neighbors of each test instance, with support for categorical and numeric features.

        Args:
            X_test (list of list of numeric or categorical values): The list of testing samples.
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances (list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test.
            neighbor_indices (list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances).
        """

        row_indexes_dists = []

        # For each test instance
        for x_test in X_test:
            for i, row in enumerate(self.X_train):
                dist = compute_distance(row, x_test)
                row_indexes_dists.append([i, dist])

        #Sort by distance (ascending)
        row_indexes_dists.sort(key=lambda x: x[1])

        # Select top k neighbors
        top_k = row_indexes_dists[:self.n_neighbors]

        nearest_distances = [top[1] for top in top_k]
        nearest_indexes = [top[0] for top in top_k]

        # Reset for next test instance
        row_indexes_dists = []

        return [nearest_distances], [nearest_indexes]

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """

        #finds the indexes and distances
        distances, indexes = self.kneighbors(X_test)
        saver = []
        saver.append(distances)
        values = []
        counts = []
        truth_value = True
        #finds the values at the indexes and counts the y_train values at the index
        for row in indexes:
            for column in row:
                #if values is zero, then adds value to values and intializes a new append to count
                if len(values) == 0:
                    values.append(self.y_train[column])
                    counts.append(1)
                else:
                    for check_status in values:
                        #finds the where the y_train ands adds a value to the count
                        if check_status == self.y_train[column]:
                            index = values.index(check_status)
                            counts[index] = counts[index] + 1
                            truth_value = False
                    #if the current value is not in the values, then it will add the values
                    if truth_value:
                        values.append(self.y_train[column])
                        counts.append(1)
                truth_value = True

        prediction = []
        max_value = counts[0]
        max_index = 0
        index = 0
        #finds the one with the most values of the choosen indexes
        for count in counts:
            if max_value < count:
                max_index = index
                max_value = count
            index = index + 1
        #adds the values to the prediction
        prediction.append(values[max_index])

        return prediction
'''

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        #finds the most common label
        values = []
        count = []
        truth_value =  0
        train_classifier = []
        train_classifier.append(X_train)
        for train_value in y_train:
            #not in the list will add a new value at the beginning
            if len(values) == 0:
                values.append(train_value)
                count.append(1)
            else:
                for x in values:
                    #if value is in the list, will add to the count
                    if x == train_value:
                        index = values.index(x)
                        count[index] = count[index] + 1
                        truth_value = False

                if truth_value:
                    values.append(train_value)
                    count.append(1)
                #if value is not in the list, then adds to the list
            truth_value = True
        #finds the value with the most label
        index = 0
        max_index  = 0
        max_value = count[0]
        for x in count:
            if max_value < x:
                max_value = x
                max_index = index
            index = index + 1
        #sets the most common label
        self.most_common_label = values[max_index]

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        #adds all the self.most_common_label for all test cases to set the prediction
        prediction = []
        for _ in range(len(X_test)):
            prediction.append(self.most_common_label)
        return prediction

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None

    def priors_calculation(self, y_train):
        """Calculates the prior probabilities for each class label.

        Args:
            y_train (list): The target values of the training data.

        Returns:
            list: A list of unique labels.
            list: A list of counts for each class label.
        """
        total_instances = len(y_train)
        total_values = []
        unique_labels = []
        for row in y_train:
            if row not in unique_labels:
                unique_labels.append(row)
                total_values.append(0)

        count = 0
        for y in unique_labels:
            for row in y_train:
                if row == y:
                    total_values[count] = total_values[count] + 1
            count = count + 1
        prior_values = []

        count = 0
        for row in total_values:

            prior_values.append([unique_labels[count], row/total_instances])
            count = count + 1
        self.priors = prior_values
        return unique_labels, total_values

    def process_x_train(self, X_train):
        """Prepares the training data for Naive Bayes by organizing the features.

        Args:
            X_train (list of list): The training instances.

        Returns:
            list: A list of unique attributes for each feature.
        """
        new_attributes = []
        unique_attributes = []
        count_test = []
        total_unique_attributes = 0
        x_train_length = len(X_train[0])
        for row in range(x_train_length):
            for y in X_train:
                if y[row] not in new_attributes:
                    new_attributes.append(y[row])
                    total_unique_attributes = total_unique_attributes + 1
            unique_attributes.append(new_attributes)
            new_attributes = []

        for row in unique_attributes:
            if isinstance(row[0], int):
                row.sort()
        for row in unique_attributes:
            for y in row:
                count_test.append(y)
        return total_unique_attributes, unique_attributes, count_test\

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """

        unique_labels, total_values = self.priors_calculation(y_train)

        total_unique_attributes, unique_attributes, count_test = self.process_x_train(X_train)

        wayward = []
        zero = []
        count = 0
        index_counter = 0
        test = []
        for row in range(total_unique_attributes):
            length_unique_labels = len(unique_labels)
            for y in range(length_unique_labels):
                zero.append(0)
            wayward.append(zero)
            if len(unique_attributes[index_counter]) == len(wayward):
                test.append(wayward)
                wayward = []
                count = 0
                index_counter = index_counter + 1
            zero = []
        test_length = len(test)
        train_length = len(X_train)
        for row in range(test_length):
            for column in range(train_length):
                value = unique_attributes[row].index(X_train[column][row])
                total = unique_labels.index(y_train[column])
                test[row][value][total] = test[row][value][total] + 1
        count = 0
        preprocessed = []
        pre = []
        row = 0

        for position in test:
            for y in position:
                preprocessed.append(count_test[row])
                for column in y:
                    preprocessed.append(column/total_values[count])
                    count = count + 1
                pre.append(preprocessed)
                count = 0
                row = row + 1
                preprocessed = []
        self.posteriors = pre

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predicted = []

        for testing_set in X_test:
            new_values = []
            for x in testing_set:
                for y in self.posteriors:
                    if y[0] == x:
                        new_values.append(y)
            pre = []
            post = []
            for x in range(len(testing_set)):
                for y in range(len(new_values[0])):
                    if y != 0:
                        pre.append(new_values[x][y])
                post.append(pre)
                pre = []

            test = []
            count = 0
            prior_length = len(self.priors)
            for x in range(prior_length):
                change = self.priors[x][1]
                for x in post:
                    change *= x[count]
                test.append(change)
                count = count + 1
            test_length = len(test)
            for i in range(test_length):
                test[i] = round(test[i], 4)

            highest = test[0]
            count = 0
            index_highest = 0
            for x in test:
                if highest < x:
                    highest = x
                    index_highest = count
                count = count + 1

            predicted.append(self.priors[index_highest][0])

        return predicted

def build_values(count, build, sub_zero, sub_one):
    """Helper function to build the values for the decision tree.

    Args:
        count (int): Counter used for the current tree level.
        build (list): List of built values so far.
        sub_zero: First value to be appended.
        sub_one: Second value to be appended.

    Returns:
        list: Updated list with new values appended.
    """
    if count == 0:
        build.append((sub_zero, sub_one))
    else:
        build.append((sub_zero, sub_one))
    return build

def preprocess_tree(subtree, count, build, previous_value = ""):
    """Recursively preprocesses a decision tree and appends values.

    Args:
        subtree (list): Subtree representing a portion of the decision tree.
        count (int): Counter for recursion depth.
        build (list): List that will hold the processed tree structure.
        previous_value (str): The previous value used to track value flow.

    Returns:
        list: Updated list representing the processed decision tree.
    """
    if subtree[0] != "Leaf":
        build = build_values(count, build, subtree[0], subtree[1])
        for node in subtree[2:]:
            count = count + 1
            build.append((node[0], node[1], previous_value))
            previous_value = node[1]
            preprocess_tree(node[2], count, build, previous_value)
        return build
    return None

def traverse_decision_tree(attribute_names, subtree, class_name, sentence = ""):
    """Recursively traverses the decision tree to print out the decision rules.

    Args:
        attribute_names (list): List of attribute names.
        subtree (list): The current subtree in the decision tree.
        class_name (str): The name of the class label.
        sentence (str): The current decision rule being built.
    """
    if subtree[0] == "Leaf":
        label = subtree[1]
        print(sentence + " THEN " + class_name + " == " + label)
        return
    if subtree[0] == "Attribute":
        attribute_name = subtree[1]
        for node in subtree[2:]:
            if node[0] == "Value":
                test = node[1]
                if sentence == "":
                    new_sentence = sentence + "IF " + attribute_name + " == " + test
                else:
                    new_sentence = sentence + " AND " + attribute_name + " == " + test
                traverse_decision_tree(attribute_names, node[2], class_name, new_sentence)

def calculate_entropy(class_counts, total_count):
    """Calculates the entropy of a given class distribution.

    Args:
        class_counts (dict): Dictionary of class label counts.
        total_count (int): Total number of instances in the dataset.

    Returns:
        float: The calculated entropy of the class distribution.
    """
    entropy = 0.0
    for count in class_counts.values():
        if count > 0:
            prob = count / total_count
            entropy -= prob * math.log(prob, 2)
    return entropy

def select_attribute(instances, attributes, attribute_domain, header):
    """Selects the best attribute to split on using the entropy-based method.

    Args:
        instances (list): List of instances (rows of data).
        attributes (list): List of available attributes to split on.
        attribute_domain (dict): Dictionary of attribute domains.
        header (list): List of attribute names.

    Returns:
        str: The attribute that provides the best split (lowest entropy).
    """
    total_instances = len(instances)
    best_attribute = None
    minimum_entropy = float('inf')
    for attribute in attributes:
        weighted_entropy = 0.0
        # Get the domain of the current attribute (possible values for the attribute)
        domain = attribute_domain[attribute]
        # For each value in the attribute's domain
        for value in domain:
            # Partition the instances based on the attribute value
            partition = [instance for instance in instances if instance[header.index(attribute)] == value]
            partition_size = len(partition)
            # Calculate the class distribution for the partition
            class_counts = {}
            for instance in partition:
                class_label = instance[-1]  # The class label is the last element in the instance
                if class_label not in class_counts:
                    class_counts[class_label] = 0
                class_counts[class_label] += 1
            # Calculate entropy for this partition
            entropy = calculate_entropy(class_counts, partition_size)
            # Calculate the weighted entropy: weight = partition size / total instances
            weighted_entropy += (partition_size / total_instances) * entropy
        # Update best attribute if we find a lower entropy
        if weighted_entropy < minimum_entropy:
            minimum_entropy = weighted_entropy
            best_attribute = attribute

    return best_attribute

def partition_instances(instances, attribute, attribute_domains, header):
    """Partitions instances based on the values of a given attribute.

    Args:
        instances (list): List of instances (rows of data).
        attribute (str): The attribute to partition the instances by.
        attribute_domains (dict): Dictionary of attribute domains.
        header (list): List of attribute names.

    Returns:
        dict: A dictionary where keys are attribute values and values are partitions of instances.
    """
    att_index = header.index(attribute)
    att_domain = attribute_domains[attribute]
    partitions = {}
    for att_value in att_domain:  # "Junior" -> "Mid" -> "Senior"
        partitions[att_value] = []
        for instance in instances:
            if instance[att_index] == att_value:
                partitions[att_value].append(instance)
    return partitions

def all_same_class(instances):
    """Checks if all instances have the same class label.

    Args:
        instances (list): List of instances (rows of data).

    Returns:
        bool: True if all instances have the same class label, False otherwise.
    """
    first_class = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_class:
            return False
    return True

def compare_instances(instances):
    """Compares instances to find the majority class label.

    Args:
        instances (list): List of instances (rows of data).

    Returns:
        tuple: The majority class label and its count.
    """
    count = {}
    for instance in instances:
        if instance[-1] not in count:
            count[instance[-1]] = 1
        else:
            count[instance[-1]] = count[instance[-1]] + 1
    max_value = 0
    class_value = 0
    for key, value in count.items():
        if value > max_value:
            max_value = value
            class_value = key
        elif value == max_value:
            alphabetical = []
            alphabetical.append(class_value)
            alphabetical.append(key)
            alphabetical.sort()
            class_value = alphabetical[0]
    return class_value, max_value

def count_current_instances(instances, att_value, att_index):
    """Counts how many times a given attribute value appears in the instances.

    Args:
        instances (list): List of instances (rows of data).
        att_value (str): The attribute value to count.
        att_index (int): The index of the attribute in the instance.

    Returns:
        int: The count of instances with the given attribute value.
    """
    count = 0
    for x in instances:
        if x[att_index] == att_value:
            count = count + 1
    return count

def tdidt(current_instances, available_attributes, attribute_domains, header, previous_size, f=1, random_seed = None, random_forest=False):
    """Recursive implementation of the TDIDT algorithm to build a decision tree.

    Args:
        current_instances (list): The current subset of instances.
        available_attributes (list): The list of available attributes for splitting.
        attribute_domains (dict): The domain of each attribute.
        header (list): The list of attribute names.
        previous_size (int): The number of instances of the previous level of the tree.

    Returns:
        list: The constructed decision tree as a nested list.
    """
    # Select the best attribute to split on
    
    if (random_forest):
        subset_attributes = compute_random_subset(available_attributes, 2, random_seed)
        split_attribute = select_attribute(current_instances, subset_attributes, attribute_domains, header)
    else:
        split_attribute = select_attribute(current_instances, available_attributes, attribute_domains, header)
    compare_instances(current_instances)
    available_attributes.remove(split_attribute)  # can't split on this attribute again
    tree = ["Attribute", split_attribute]
    # Group data by attribute values
    partitions = partition_instances(current_instances, split_attribute, attribute_domains, header)

    for att_value in sorted(partitions.keys()):  # process in alphabetical order
        att_partition = partitions[att_value]
        value_subtree = ["Value", att_value]
        # Case 1: All class labels of the partition are the same -> make a leaf node
        if len(att_partition) > 0 and all_same_class(att_partition):
            column_name = header.index(split_attribute)
            value_subtree.append(["Leaf", att_partition[0][-1],
                                  count_current_instances(current_instances, att_value, column_name), len(current_instances)])
            tree.append(value_subtree)
        # Case 2: No more attributes to select -> make a majority vote leaf node
        elif len(att_partition) > 0 and len(available_attributes) == 0:
            majority_class, total_count = compare_instances(att_partition)
            value_subtree.append(["Leaf", majority_class, total_count, len(current_instances)])
            tree.append(value_subtree)
        # Case 3: No instances in the partition -> backtrack and make a majority vote leaf node
        elif len(att_partition) == 0:
            majority_class, total_count = compare_instances(current_instances)
            tree = ["Leaf", majority_class, len(current_instances), previous_size]
            return tree
        else:
            previous_size = len(current_instances)
            # Recursively build the subtree for this partition
            subtree = tdidt(att_partition, available_attributes.copy(), attribute_domains, header, previous_size, random_forest=random_forest)
            value_subtree.append(subtree)
            tree.append(value_subtree)   
    return tree

def tdidt_predict(tree, instance, header):
    """Makes a prediction for a single instance using the decision tree.

    Args:
        tree (list): The decision tree.
        instance (list): The instance to predict.
        header (list): The list of attribute names.

    Returns:
        str: The predicted class label.
    """
    if tree[0] == "Leaf":
        return tree[1]
    att_index = header.index(tree[1])
    for i in range(2, len(tree)):
        value_list = tree[i]
        if value_list[1] == instance[att_index]:
            return tdidt_predict(value_list[2], instance, header)
    return None

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
        y_train(list of obj): The target y values (parallel to X_train).
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier.
    """

    def __init__(self, random_seed = None):
        """Initializer for MyDecisionTreeClassifier."""
        self.X_train = None
        self.y_train = None
        self.tree = None
        self.random_seed = random_seed
        

    def fit(self, X_train, y_train, f=1, random_forest=False):
        """Fits a decision tree classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
            y_train(list of obj): The target y values (parallel to X_train).
        """
        self.X_train = X_train
        self.y_train = y_train
        attribute_label = "att"
        header = []
        for x in range(len(X_train[0])):
            full = attribute_label + str(x)
            header.append(full)
        attribute_domains = {}
        unique = []
        header_length = len(header)
        for x in range(header_length):
            for y in X_train:
                if y[x] not in unique:
                    unique.append(y[x])
            attribute_domains[header[x]] = unique
            unique = []
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        available_attributes = header.copy()

        tree = tdidt(train, available_attributes, attribute_domains, header, 0, random_forest=random_forest)
        self.tree = tree

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples.

        Returns:
            list of obj: The predicted target y values (parallel to X_test).
        """
        attribute_label = "att"
        header = []
        for x in range(len(self.X_train[0])):
            full = attribute_label + str(x)
            header.append(full)
        predictions = []
        for instance in X_test:
            prediction = tdidt_predict(self.tree, instance, header)
            predictions.append(prediction)
        return predictions

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree.

        Args:
            attribute_names(list of str): A list of attribute names.
            class_name(str): The name of the class label.
        """
        if attribute_names is None:
            attribute_label = "att"
            attribute_names = []
            for x in range(len(self.X_train[0])):
                full = attribute_label + str(x)
                attribute_names.append(full)
        traverse_decision_tree(attribute_names, self.tree, class_name, sentence = "")
    
    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """Visualizes a tree using Graphviz.

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file.
            attribute_names(list of str): A list of attribute names.
        """
        print(dot_fname, attribute_names, pdf_fname)

# class MyDecisionTreeClassifier: # Charlie's Version
#     """Represents a decision tree classifier.

#     Attributes:
#         X_train(list of list of obj): The list of training instances (samples).
#                 The shape of X_train is (n_train_samples, n_features)
#         y_train(list of obj): The target y values (parallel to X_train).
#             The shape of y_train is n_samples
#         header(list of obj): list of strings representing each attribute in X_train
#         domain(dictionary): A dictionary of all corresponding values for each attribute
#         tree(nested list): The extracted tree model.
#         y_labels(list of obj): A list of each unique possible target y value

#     Notes:
#         Loosely based on sklearn's DecisionTreeClassifier:
#             https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
#         Terminology: instance = sample = row and attribute = feature = column
#     """
#     def __init__(self):
#         """Initializer for MyDecisionTreeClassifier.
#         """
#         self.X_train = None
#         self.y_train = None
#         self.header = None
#         self.domain = None
#         self.tree = None
#         self.y_labels = None

#     def fit(self, X_train, y_train):
#         """Fits a decision tree classifier to X_train and y_train using the TDIDT
#         (top down induction of decision tree) algorithm.

#         Args:
#             X_train(list of list of obj): The list of training instances (samples).
#                 The shape of X_train is (n_train_samples, n_features)
#             y_train(list of obj): The target y values (parallel to X_train)
#                 The shape of y_train is n_train_samples

#         Notes:
#             Since TDIDT is an eager learning algorithm, this method builds a decision tree model
#                 from the training data.
#             Build a decision tree using the nested list representation described in class.
#             On a majority vote tie, choose first attribute value based on attribute domain ordering.
#             Store the tree in the tree attribute.
#             Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
#         """

#         self.X_train = X_train
#         self.y_train = y_train
#         self.domain = {}
#         attribute_totals = {}
#         self.header = []
#         for i in range(len(self.X_train[0])): # building self.header and self.domain
#             attribute = f"att{i}"
#             self.header.append(attribute)
#             new_row = []
#             attribute_totals[attribute] = {}
#             for row in X_train:
#                 if row[i] not in new_row:
#                     new_row.append(row[i])
#                     total = 0
#                     for instance in X_train:
#                         if row[i] in instance:
#                             total += 1
#                     attribute_totals[attribute][row[i]] = total
#             self.domain[attribute] = new_row

#         self.y_labels = []
#         for label in y_train:
#             if label not in self.y_labels:
#                 self.y_labels.append(label)

#         train = [X_train[i] + [y_train[i]] for i in range(len(X_train))] # stitching together X_train and y_train
#         #print(train)
#         #print(self.header)
#         #print(self.domain)
#         #print("attribute totals: ", attribute_totals)
        
#         # make a copy a header, b/c python is pass by object reference
#         # and tdidt will be removing attributes from available_attributes
#         available_attributes = self.header.copy()
#         used_attributes = []
#         used_instances = []

#         self.tree = myutils.tdidt(train, available_attributes, self.header, self.domain, self.y_labels, used_attributes, used_instances)
        
#         pass
        

#     def predict(self, X_test):
#         """Makes predictions for test instances in X_test.

#         Args:
#             X_test(list of list of obj): The list of testing samples
#                 The shape of X_test is (n_test_samples, n_features)

#         Returns:
#             y_predicted(list of obj): The predicted target y values (parallel to X_test)
#         """
#         y_predicted = []
#         for instance in X_test:
#             y_predicted.append(myutils.tdidt_predict(self.tree, instance, self.header))
#         return y_predicted

#     def print_decision_rules(self, attribute_names=None, class_name="class"):
#         """Prints the decision rules from the tree in the format
#         "IF att == val AND ... THEN class = label", one rule on each line.

#         Args:
#             attribute_names(list of str or None): A list of attribute names to use in the decision rules
#                 (None if a list is not provided and the default attribute names based on indexes
#                 (e.g. "att0", "att1", ...) should be used).
#             class_name(str): A string to use for the class name in the decision rules
#                 ("class" if a string is not provided and the default name "class" should be used).
#         """

#         values = list(self.domain.values())
#         combinations = myutils.generate_combinations(values)
#         #print(combinations)
#         if (attribute_names):
#             att_names = attribute_names
#         else: 
#             att_names = self.header.copy()
        
#         for row in combinations:
#             new_rule = " "
#             for att_index in range(len(row)):
#                 if att_index == 0:
#                     new_rule += f"IF {att_names[att_index]} == {row[att_index]} "
#                 else:
#                     new_rule += f"AND {att_names[att_index]} == {row[att_index]} "
#             prediction = self.predict([row])
#             new_rule += f"THEN {class_name} == {prediction[0]}"
#             print(new_rule)

#         pass



def compute_bootstrapped_sample(table):
    n = len(table)
    # np.random.randint(low, high) returns random integers from low (inclusive) to high (exclusive)
    sampled_indexes = [np.random.randint(0, n) for _ in range(n)]
    sample = [table[index] for index in sampled_indexes]
    out_of_bag_indexes = [index for index in list(range(n)) if index not in sampled_indexes]
    out_of_bag_sample = [table[index] for index in out_of_bag_indexes]
    return sample, out_of_bag_sample

def randomize_in_place(alist, parallel_list=None, random_state=None):
    """Randomizes the order of elements in a list in place.

    Args:
        alist (list): The list of elements to be randomized.
        parallel_list (list, optional): A second list to randomize in parallel with alist. Defaults to None.

    Returns:
        None: The function modifies alist and parallel_list in place, if provided.
    """
    np.random.seed(random_state)
    alist_length = len(alist)
    for i in range(alist_length):
        # generate a random index to swap this value at i with
        rand_index = np.random.randint(0, len(alist))  # rand int in [0, len(alist))
        # do the swap
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] = parallel_list[rand_index], parallel_list[i]
def bin_sizes(n_splits, randomized_x):
    """Calculates the sizes of bins for k-fold cross-validation.

    Args:
        n_splits (int): The number of splits (folds) for cross-validation.
        randomized_x (list): The input data.

    Returns:
        bins (list of list): A list of lists where each inner list contains the size of each bin.
    """
    bins = [[] for _ in range(n_splits)]

    total_size = len(randomized_x)
    elements_per_bin = int(total_size / n_splits)
    extra_index = total_size % n_splits
    num_elements = 0

    for bin_location in range(n_splits):
        if bin_location < extra_index:
            num_elements = elements_per_bin + 1
        else:
            num_elements = elements_per_bin
        bins[bin_location].append(num_elements)
    return bins

def distribution_and_labels(randomized_x, randomized_y):
    """Generates the distribution and labels from randomized data.

    Args:
        randomized_x (list): The input data features.
        randomized_y (list): The randomized labels.

    Returns:
        distribution (list): The distribution of labels as proportions.
        labels (list): The unique labels found in randomized_y.
        folds (list): A list of indices representing the folds.
    """
    y_values = []
    labels = []
    distribution = []
    folds = []

    for x in range(len(randomized_x)):
        folds.append(x)
        y_values.append(randomized_y[x])
        if randomized_y[x] not in labels:
            labels.append(randomized_y[x])
            distribution.append(1)
        else:
            total_index = labels.index(randomized_y[x])
            distribution[total_index] += 1

    distribution_length = len(distribution)
    for x in range(distribution_length):
        distribution[x] = distribution[x] / len(randomized_y)
    return distribution, labels, folds

def get_labels(labels, y_value):
    """Retrieves the index of a label from the list of labels.

    Args:
        labels (list): The list of unique labels.
        y_value: The label to find.

    Returns:
        int: The index of the label in the `labels` list.
    """
    label_index = 0
    labels_length = len(labels)
    for x in range(labels_length):
        if labels[x] == y_value:
            label_index = x
    return label_index

def even_stratified_k_fold(test, folds, labels, index_test, randomized_y, expected_number, bin_index):
    """Performs an even stratified k-fold selection for cross-validation.

    Args:
        test (list): The list to append selected test indices.
        folds (list): The list of available indices to select from.
        labels (list): The list of labels corresponding to the data.
        index_test (list): The list to track indices of the test data.
        randomized_y (list): The randomized labels.
        expected_number (list): The expected number of instances per label.
        bin_index (int): The current bin index for stratification.

    Returns:
        None: The function modifies the input lists in place.
    """
    if bin_index % 2 == 0:
        for test_index in folds:
            labels_length = len(labels)
            if len(test) == 0:
                test.append(test_index)
                index_test.append(test_index)
                for x in range(labels_length):
                    if labels[x] == randomized_y[test_index]:
                        expected_number[x] -= 1
            elif test_index not in test:
                x = get_labels(labels, randomized_y[test_index])
                if expected_number[x] > 0:
                    test.append(test_index)
                    index_test.append(test_index)
                    expected_number[x] -= 1
def distribution_values(distribution, bin_index):
    """Calculates expected values based on a distribution and bin index.

    Args:
        distribution (list of float): A list representing the distribution of values.
        bin_index (int): The index of the bin used to calculate expected values.

    Returns:
        expected_number (list): A list of expected values for each distribution element.
        current_distribution (list): A list initialized to zero for current distribution tracking.
    """
    expected_number = []
    current_distribution = []
    for x in distribution:
        estimated_values = x * bin_index
        expected_number.append(estimated_values)
        current_distribution.append(0)
    return expected_number, current_distribution

def connects_all_values(all_tests, folds):
    """Connects all test values with remaining fold indices.

    Args:
        all_tests (list): A list of lists where each inner list contains selected test indices.
        folds (list): A list of all fold indices.

    Returns:
        rows (list): A list of connections between all test values and remaining fold indices.
    """
    skils_list = []
    rows = []
    true = []
    for x in all_tests:
        for bin_values in folds:
            if bin_values not in x:
                skils_list.append(bin_values)
        true.append(skils_list)
        true.append(x)
        rows.append(true)
        true = []
        skils_list = []
    return rows

def check_add_value(index_test, bin_index, current_distribution, previous_distribution, labels, test_y_value):
    """Checks if adding a new value maintains the distribution balance.

    Args:
        index_test (list): The list of indices currently in the test set.
        bin_index (int): The current bin index.
        current_distribution (list): The current distribution of labels.
        previous_distribution (list): The previous distribution of labels.
        labels (list): The unique labels.
        test_y_value: The label of the new test value being added.

    Returns:
        bool: True if the value can be added without violating distribution constraints, False otherwise.
    """
    temp = []
    b = True
    labels_length = len(labels)
    if len(index_test) + 1 == bin_index:
        for x in range(labels_length):
            if labels[x] == test_y_value:
                updated_distribution = current_distribution[x] + 1
                temp.append(updated_distribution)
            else:
                temp.append(current_distribution[x])
    if len(previous_distribution) > 0 and previous_distribution == temp:
        b = False
    return b

def size_test_zero(test, labels, randomized_y, test_index, index_test, expected_number, current_distribution):
    """Handles the case where the test set starts empty by adding the first value.

    Args:
        test (list): The list to append the first test index.
        labels (list): The list of unique labels.
        randomized_y (list): The randomized labels.
        test_index (int): The index of the test value to add.
        index_test (list): The list tracking indices of the test data.
        expected_number (list): The expected number of instances per label.
        current_distribution (list): The current distribution of labels.

    Returns:
        None: The function modifies the input lists in place.
    """
    labels_length = len(labels)
    test.append(test_index)
    index_test.append(test_index)
    for x in range(labels_length):
        if labels[x] == randomized_y[test_index]:
            expected_number[x] -= 1
            current_distribution[x] += 1

def stratified_test_and_remainder(X, y, random_state=None, shuffle=False):
    """Split dataset into stratified cross-validation folds.

    Args:
        X (list of list of obj): The list of instances (shape: n_samples, n_features)
        y (list of obj): The target y values (parallel to X) (shape: n_samples)
        random_state (int): Seed for random number generator for reproducible results
        shuffle (bool): Randomize order of instances before creating folds

    Returns:
        folds (list of 2-item tuples): List of folds (training and testing indices)
    """
    np.random.seed(random_state)  # Seed for reproducibility
    randomized_x = X[:]  # Create a copy of X
    randomized_y = y[:]  # Create a copy of y

    # Get distribution and labels for stratification
    distribution, labels, folds = distribution_and_labels(randomized_x, randomized_y)

    if shuffle:
        # Randomize the order of folds if shuffle is True
        randomize_in_place(folds, randomized_y)
    print(randomized_y)
    # Create bins for stratified splitting
    
    index_test = []
    test = []
    expected_number = []
    all_tests = []
    previous_distribution = []
    current_distribution = []
    print(len(X))
    test_size = round((len(X) * 0.33) + 1)
    print(test_size)  
    print(len(test))
    previous_test_sizes = [] 
    # Calculate expected distribution of classes in each bin
    expected_number, current_distribution = distribution_values(distribution, test_size)
    test_size = round((len(X) * 0.33) + 1)
    even_stratified_k_fold(test, folds, labels, index_test, randomized_y, expected_number, test_size)
    # Handle odd-sized bins to ensure balance
    if test_size % 2 != 0:
        print(test_size)
        count = 0
        for _ in range(test_size):
            valid = True
            while valid:
                current_random_index = np.random.randint(len(X), size=1)
                if current_random_index not in previous_test_sizes:
                    valid = False
                    previous_test_sizes.append(current_random_index)
            
            count = count + 1
            pass
            if len(test) == 0:
                size_test_zero(test, labels, randomized_y, current_random_index[0], index_test, expected_number, current_distribution)
            elif check_add_value(index_test, test_size, current_distribution, previous_distribution,
                                            labels, randomized_y[current_random_index[0]]) and current_random_index not in test:
                x = get_labels(labels, randomized_y[current_random_index[0]])
                if expected_number[x] > 1:
                    current_distribution[x] += 1
                    test.append(current_random_index)
                    index_test.append(current_random_index[0])
                    expected_number[x] -= 1
                elif 0 < expected_number[x] <= 1 and len(index_test) < test_size:
                    current_distribution[x] += 1
                    test.append(current_random_index)
                    index_test.append(current_random_index[0])
            
        previous_distribution = current_distribution

    all_tests.append(index_test)
    index_test = []
    expected_number = []
    # Connect all test indices with their corresponding folds
    
    rows = connects_all_values(all_tests, folds)
    y_values = []
    y_hold = []
        
    for x in rows:
        for n in x:
            for z in range(len(n)):
                n[z] = X[n[z]]
                y_hold.append(y[z])
            y_values.append(y_hold)
            y_hold =[]

    remainder = rows[0][0]
    test_set = rows[0][1]

    return remainder, test_set, y_values[0], y_values[1]

def compute_random_subset(values, num_values):
    values_copy = values[:] # shallow copy
    np.random.shuffle(values_copy) # in place shuffle
    return values_copy[:num_values]

def compute_bootstrapped_sample(table, y):
    n = len(table)
    training_set = round(len(table) * 0.63 + 1)
    # np.random.randint(low, high) returns random integers from low (inclusive) to high (exclusive)
    sampled_indexes = [np.random.randint(0, n) for _ in range(training_set)]
    sample = [table[index] for index in sampled_indexes]
    y_samples = [y[index] for index in sampled_indexes]

    out_of_bag_indexes = [index for index in range(n) if index not in sampled_indexes]
    out_of_bag_sample = [table[index] for index in out_of_bag_indexes]
    y_out_of_bag_sample = [y[index] for index in out_of_bag_indexes]
    
    return sample, out_of_bag_sample, y_samples, y_out_of_bag_sample

def classifierAccuracy(prediction, actual_prediction):
    total_count = 0
    actual_count = 0
    for x in range(len(prediction[0])):
        if prediction[0][x] == actual_prediction[x]:
            actual_count = actual_count + 1
        total_count = total_count + 1
    accuracy = actual_count/total_count
    return accuracy
def find_best_classifiers(decision_accuracy, forest, M_size):
    best_classifiers = []
    max_accuracy = decision_accuracy[0]
    max_tree = forest[0]
    max_index = 0
    for x in range(M_size):
        for y in range(len(forest)):
            if (decision_accuracy[y] > max_accuracy):
                max_accuracy = decision_accuracy[y]
                max_tree = forest[y]
                max_index = y
        forest.pop(max_index)
        decision_accuracy.pop(max_index)
        best_classifiers.append(max_tree)
        max_index = 0
        max_accuracy = decision_accuracy[0]
        max_tree = forest[0]
    forest = best_classifiers
    return best_classifiers
def setTests(table, class_labels, test_instances):
    train = []
    y_train = []
    test = []
    y_test = []
    for x in test_instances[0]:
        for y in x:
            train.append(table[y])
            y_train.append(class_labels[y])
    for x in test_instances[1]:
        for y in x:
            test.append(table[y])
            y_test.append(class_labels[y])
    return train, test, y_train, y_test
class MyRandomForestClassifier: 
    def __init__(self, random_seed = None ):
        """ Intializer of the My"""
        self.X_train = None
        self.y_train = None
        self.forest = None
        self.random_seed = random_seed
        self.performance = None
    
    def fit(self, X, y, n_samples = 10, f=2, M=1):
        test = myevaluation.stratified_kfold_split(X, y, n_splits=3, random_state=0, shuffle=True)
        remainder, test_set, y_remainder, y_test = setTests(X, y, test)
        self.forest = []
        decision_accuracy = []
        unique_labels = []
        for x in y:
            if x not in unique_labels:
                unique_labels.append(x)
        for x in range(n_samples):
            training_set, validation, y_training, y_validation = compute_bootstrapped_sample(remainder, y_remainder)

            decision_tree = MyDecisionTreeClassifier(random_seed=self.random_seed)
            decision_tree.fit(training_set,y_training, f)
            prediction = []
            prediction.append(decision_tree.predict(validation))
            count = 0
            total_count = 0
            for x in range(len(prediction[0])):
                if prediction[0][x] == y_validation[x]:
                    count = count + 1
                total_count = total_count + 1            
            decision_accuracy.append(myevaluation.accuracy_score(y_validation, prediction[0]))
            self.forest.append(decision_tree)
        #Finds the best classifiers
        self.forest = find_best_classifiers(decision_accuracy, self.forest, M)
        total_predictions = self.predict(test_set)
        #finds the total performance of the classifier
        self.performance = myevaluation.accuracy_score(y_test, total_predictions)
        count = 0
        for x in total_predictions:
            if x == 'A':
                count = count + 1 
    def predict(self, X_test):
        predictions = []
        m_size = len(self.forest)
        for x in self.forest:
           predictions.append(x.predict(X_test))
        unique_label = []
        count = []
        max_prediction = ""
        max_count = 0
        total_predictions = []
        
        for x in range(len(predictions[0])):
            for y in range(m_size):
                if predictions[y][x] not in unique_label:
                    unique_label.append(predictions[y][x])
                    count.append(1)
                else:
                    index_value = unique_label.index(predictions[y][x])
                    count[index_value] = count[index_value] + 1
                    if count[index_value] > max_count:
                        max_prediction = predictions[y][x]
                        max_count = count[index_value]
            total_predictions.append(max_prediction)
            max_count = 0
            unique_label = []
            count = []
        
        return total_predictions