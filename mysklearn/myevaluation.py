import numpy as np
from mysklearn import myutils

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

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X (list of list of obj): The list of samples (shape: n_samples, n_features)
        y (list of obj): The target y values (parallel to X) (shape: n_samples)
        test_size (float or int): Proportion or absolute number of instances for the test set
        random_state (int): Seed for random number generator for reproducible results
        shuffle (bool): Randomize order of instances before splitting

    Returns:
        X_train (list of list of obj): Training samples
        X_test (list of list of obj): Testing samples
        y_train (list of obj): Target y values for training (parallel to X_train)
        y_test (list of obj): Target y values for testing (parallel to X_test)

    Note:
        Based on sklearn's train_test_split().
    """
    randomized_x = X[:]  # Create a copy of X
    randomized_y = y[:]  # Create a copy of y
    n = len(randomized_x)  # Number of samples

    if shuffle:
        # Randomize the order of samples if shuffle is True
        randomize_in_place(randomized_x, randomized_y, random_state)

    # Determine split index based on test_size
    if test_size < 1:
        split_index = int((1 - test_size) * n)
    else:
        split_index = int(n - test_size)

    # Return the split datasets
    return randomized_x[0:split_index], randomized_x[split_index:], randomized_y[0:split_index], randomized_y[split_index:]

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

def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross-validation folds.

    Args:
        X (list of list of obj): The list of samples (shape: n_samples, n_features)
        n_splits (int): Number of folds
        random_state (int): Seed for random number generator for reproducible results
        shuffle (bool): Randomize order of instances before creating folds

    Returns:
        folds (list of 2-item tuples): List of folds (training and testing indices)
    """
    np.random.seed(random_state)  # Seed for reproducibility
    randomized_x = X[:]  # Create a copy of X
    n = len(randomized_x)  # Number of samples
    folds = list(range(n))  # Initialize folds with sample indices

    if shuffle:
        # Randomize the order of folds if shuffle is True
        randomize_in_place(folds)

    # Create bins for cross-validation
    bins = [[] for _ in range(n_splits)]
    total_size = len(randomized_x)
    elements_per_bin = total_size // n_splits
    extra_index = total_size % n_splits
    indicies = 0

    # Fill bins with samples
    for bin_location in range(n_splits):
        num_elements = elements_per_bin + 1 if bin_location < extra_index else elements_per_bin
        for _ in range(num_elements):
            if indicies < n:
                bins[bin_location].append(folds[indicies])
                indicies += 1

    # Prepare the training and testing indices for each fold
    skils_list = []
    rows = []
    true = []

    for x in bins:
        for fold_values in folds:
            if fold_values not in x:
                skils_list.append(fold_values)
        true.append(skils_list)
        true.append(x)
        rows.append(true)
        true = []
        skils_list = []

    return rows

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

def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross-validation folds.

    Args:
        X (list of list of obj): The list of instances (shape: n_samples, n_features)
        y (list of obj): The target y values (parallel to X) (shape: n_samples)
        n_splits (int): Number of folds
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

    # Create bins for stratified splitting
    bins = bin_sizes(n_splits, randomized_x)

    index_test = []
    test = []
    expected_number = []
    all_tests = []
    previous_distribution = []
    current_distribution = []

    for bin_values in bins:
        # Calculate expected distribution of classes in each bin
        expected_number, current_distribution = distribution_values(distribution, bin_values[0])
        even_stratified_k_fold(test, folds, labels, index_test, randomized_y, expected_number, bin_values[0])
        # Handle odd-sized bins to ensure balance
        if bin_values[0] % 2 != 0:
            for test_index in folds:
                if len(test) == 0:
                    size_test_zero(test, labels, randomized_y, test_index, index_test, expected_number, current_distribution)
                elif check_add_value(index_test, bin_values[0], current_distribution, previous_distribution,
                                             labels, randomized_y[test_index]) and test_index not in test:
                    x = get_labels(labels, randomized_y[test_index])
                    if expected_number[x] > 1:
                        current_distribution[x] += 1
                        test.append(test_index)
                        index_test.append(test_index)
                        expected_number[x] -= 1
                    elif 0 < expected_number[x] <= 1 and len(index_test) < bin_values[0]:
                        current_distribution[x] += 1
                        test.append(test_index)
                        index_test.append(test_index)
            previous_distribution = current_distribution
        all_tests.append(index_test)
        index_test = []
        expected_number = []

    # Connect all test indices with their corresponding folds
    rows = connects_all_values(all_tests, folds)

    return rows

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out-of-bag test set.

    Args:
        X (list of list of obj): The list of samples
        y (list of obj): The target y values (parallel to X) (default is None)
        n_samples (int): Number of samples to generate (default is len(X))
        random_state (int): Seed for random number generator for reproducible results

    Returns:
        X_sample (list of list of obj): Sampled training set
        X_out_of_bag (list of list of obj): Out-of-bag samples
        y_sample (list of obj): Sampled target y values (parallel to X_sample)
        y_out_of_bag (list of obj): Out-of-bag target y values (parallel to X_out_of_bag)
    """
    np.random.seed(random_state)  # Seed for reproducibility
    if n_samples is None:
        n_samples = len(X)  # Default to the size of X

    X_sample = []
    y_sample = []
    X_out_of_bag = []
    y_out_of_bag = []
    size_x = len(X)
    size_y = len(y)

    # Store indices of sampled instances
    storeindex = []
    if y is None:
        y_sample = []  # No sampling for y if it is None

    if y is not None:
        # Sample instances with replacement
        for _ in range(n_samples):
            j = np.random.randint(0, size_x)
            X_sample.append(X[j])
            y_sample.append(y[j])
            storeindex.append(j)

        # Collect out-of-bag samples
        for x in range(size_x):
            if x not in storeindex:
                X_out_of_bag.append(X[x])
        for fold_values in range(size_y):
            if fold_values not in storeindex:
                y_out_of_bag.append(y[fold_values])
    else:
        # Handle case where y is None
        for _ in range(n_samples):
            j = np.random.randint(0, size_x)
            X_sample.append(X[j])
            if X[j] not in X_out_of_bag:
                X_out_of_bag.append(X[j])

    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true (list of obj): Ground truth target y values (shape: n_samples)
        y_pred (list of obj): Predicted target y values (parallel to y_true) (shape: n_samples)
        labels (list of str): List of all possible target y labels used to index the matrix

    Returns:
        matrix (list of list of int): Confusion matrix indicating counts of true vs predicted labels
    """
    y_actual = []
    y_prediction = []
    label_length = len(labels)
    y_true_length = len(y_true)

    # Convert true and predicted values to indices based on labels
    if isinstance(y_true[0], str):
        for value in range(y_true_length):
            string_index = labels.index(y_true[value])
            y_actual.append(string_index)
            string_index = labels.index(y_pred[value])
            y_prediction.append(string_index)
    else:
        y_actual = y_true
        y_prediction = y_pred

    # Initialize the confusion matrix
    matrix_value = [[0 for _ in range(label_length)] for _ in range(label_length)]

    # Fill the confusion matrix
    for x in range(y_true_length):
        if y_actual[x] == y_prediction[x]:
            value = y_actual[x]
            matrix_value[value][value] += 1
        else:
            value = y_actual[x]
            predicted_y = y_prediction[x]
            matrix_value[value][predicted_y] += 1

    return matrix_value

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true (list of obj): Ground truth target y values (shape: n_samples)
        y_pred (list of obj): Predicted target y values (parallel to y_true) (shape: n_samples)
        normalize (bool): If False, return the number of correctly classified samples; otherwise, return the fraction

    Returns:
        score (float): Fraction or count of correctly classified samples
    """
    correct = sum(1 for x in range(len(y_true)) if y_true[x] == y_pred[x])  # Count correct predictions
    if normalize:
        return correct / len(y_true)  # Return fraction of correct predictions
    return correct  # Return count of correct predictions


def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    true_positive = 0
    false_positive = 0
    if labels is None:
        labels = []
        for x in y_true:
            if x not in labels:
                labels.append(x)
    if pos_label is None:
        pos_label = labels[0]
    y_length = len(y_true)
    for y in range(y_length):
        if pos_label == y_pred[y]:
            if y_true[y] == y_pred[y]:
                true_positive = true_positive + 1
            else:
                false_positive = false_positive + 1
    precision = 0.0
    if false_positive + true_positive != 0:
        precision = true_positive/(true_positive + false_positive)

    return precision


def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """

    true_positive = 0
    false_negative = 0
    if labels is None:
        labels = []
        for x in y_true:
            if x not in labels:
                labels.append(x)
    if pos_label is None:
        pos_label = labels[0]
    y_length = len(y_true)
    for y in range(y_length):
        if pos_label == y_pred[y]:
            if y_true[y] == y_pred[y]:
                true_positive = true_positive + 1
        else:
            if y_true[y] == pos_label:
                false_negative = false_negative + 1
    recall = 0.0
    if false_negative + true_positive != 0:
        recall = true_positive/(true_positive + false_negative)

    return recall

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    precision = binary_precision_score(y_true, y_pred, labels=labels, pos_label=pos_label)
    recall = binary_recall_score(y_true, y_pred, labels=labels, pos_label=pos_label)
    f1 = 0
    if precision + recall != 0:
        f1 = 2 * (precision * recall)/(precision + recall)
    return f1
