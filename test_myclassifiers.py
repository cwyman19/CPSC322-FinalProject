from mysklearn.myclassifiers import MyRandomForestClassifier

# note: order is actual/received student value, expected/solution
def test_random_forest_classifier_fit():
    X_train = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
    cluster_test = MyRandomForestClassifier()
    cluster_test.fit(X_train, y_train, n_samples = 5 , f=2, M=3)
    assert 3 == len(cluster_test.forest)
    assert cluster_test.forest[0] != cluster_test.forest[1]
    assert cluster_test.forest[0] != cluster_test.forest[2]
    assert cluster_test.forest[1] != cluster_test.forest[2]

    X_train = [
    [1, 3, 'fair'],
    [1, 3, 'excellent'],
    [2, 3, 'fair'],
    [2, 2, 'fair'],
    [2, 1, 'fair'],
    [2, 1, 'excellent'],
    [2, 1, 'excellent'],
    [1, 2, 'fair'],
    [1, 1, 'fair'],
    [2, 2, 'fair'],
    [1, 2, 'excellent'],
    [2, 2, 'excellent'],
    [2, 3, 'fair'],
    [2, 2, 'excellent'],
    [2, 3, 'fair']
]
    y_train =  ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'yes']
    cluster_test = MyRandomForestClassifier()
    cluster_test.fit(X_train, y_train, n_samples = 5 , f=2, M=3)
    assert 3 == len(cluster_test.forest)
    assert cluster_test.forest[0] != cluster_test.forest[1]
    assert cluster_test.forest[0] != cluster_test.forest[2]
    assert cluster_test.forest[1] != cluster_test.forest[2]


def test_random_forest_classifier_predict():
    X_train = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
    cluster_test = MyRandomForestClassifier()
    cluster_test.fit(X_train, y_train, n_samples = 5 , f=2, M=3)
    X_test = [['Junior', 'Java', 'yes', 'no'], ['Junior', 'Java', 'yes', 'yes']]
    X_actual = ["True", "True"]
    print(cluster_test.forest[0].tree)
    print(cluster_test.forest[1].tree)
    print(cluster_test.forest[2].tree)
    cluster_test.forest[0].tree = ['Attribute', 'att0', ['Value', 'Junior', ['Attribute', 'att2', ['Value', 'no', ['Leaf', 'False', 1, 3]], ['Value', 'yes', ['Leaf', 'True', 2, 3]]]], ['Value', 'Mid', ['Leaf', 'True', 1, 10]], ['Value', 'Senior', ['Attribute', 'att2', ['Value', 'no', ['Leaf', 'False', 2, 6]], ['Value', 'yes', ['Leaf', 'True', 4, 6]]]]]
    cluster_test.forest[1].tree = ['Attribute', 'att3', ['Value', 'no', ['Leaf', 'True', 4, 10]], ['Value', 'yes', ['Attribute', 'att0', ['Value', 'Junior', ['Leaf', 'False', 3, 6]], ['Value', 'Mid', ['Leaf', 'True', 2, 6]], ['Value', 'Senior', ['Leaf', 'False', 1, 6]]]]]
    cluster_test.forest[2].tree = ['Attribute', 'att0', ['Value', 'Junior', ['Leaf', 'True', 1, 10]], ['Value', 'Mid', ['Leaf', 'True', 6, 10]], ['Value', 'Senior', ['Leaf', 'False', 3, 10]]]
    prediction = cluster_test.predict(X_test)
    assert X_actual == prediction
    
    X_train = [
    [1, 3, 'fair'],
    [1, 3, 'excellent'],
    [2, 3, 'fair'],
    [2, 2, 'fair'],
    [2, 1, 'fair'],
    [2, 1, 'excellent'],
    [2, 1, 'excellent'],
    [1, 2, 'fair'],
    [1, 1, 'fair'],
    [2, 2, 'fair'],
    [1, 2, 'excellent'],
    [2, 2, 'excellent'],
    [2, 3, 'fair'],
    [2, 2, 'excellent'],
    [2, 3, 'fair']
    ]
    y_train =  ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'yes']
    cluster_test = MyRandomForestClassifier()
    cluster_test.fit(X_train, y_train, n_samples = 5 , f=2, M=3)
    cluster_test.forest[0].tree = ['Attribute', 'att1', ['Value', 1, ['Leaf', 'yes', 2, 10]], ['Value', 2, ['Attribute', 'att0', ['Value', 1, ['Leaf', 'no', 1, 5]], ['Value', 2, ['Attribute', 'att2', ['Value', 'excellent', ['Leaf', 'no', 2, 4]], ['Value', 'fair', ['Leaf', 'yes', 1, 4]]]]]], ['Value', 3, ['Attribute', 'att0', ['Value', 1, ['Leaf', 'no', 2, 3]], ['Value', 2, ['Leaf', 'yes', 1, 3]]]]]
    cluster_test.forest[1].tree = ['Attribute', 'att0', ['Value', 1, ['Attribute', 'att1', ['Value', 1, ['Leaf', 'yes', 1, 5]], ['Value', 2, ['Attribute', 'att2', ['Value', 'excellent', ['Leaf', 'yes', 2, 3]], ['Value', 'fair', ['Leaf', 'no', 1, 3]]]], ['Value', 3, ['Leaf', 'no', 1, 5]]]], ['Value', 2, ['Leaf', 'yes', 5, 10]]]
    cluster_test.forest[2].tree = ['Attribute', 'att1', ['Value', 1, ['Leaf', 'yes', 2, 10]], ['Value', 2, ['Attribute', 'att0', ['Value', 1, ['Attribute', 'att2', ['Value', 'excellent', ['Leaf', 'yes', 1, 3]], ['Value', 'fair', ['Leaf', 'no', 2, 3]]]], ['Value', 2, ['Attribute', 'att2', ['Value', 'excellent', ['Leaf', 'no', 1, 3]], ['Value', 'fair', ['Leaf', 'yes', 2, 3]]]]]], ['Value', 3, ['Attribute', 'att0', ['Value', 1, ['Leaf', 'no', 1, 2]], ['Value', 2, ['Leaf', 'yes', 1, 2]]]]]
    X_test = [[2, 2, 'fair'], [2, 1, 'excellent']]
    X_actual = ["yes", "yes"]
    prediction = cluster_test.predict(X_test)
    print(prediction)
    assert X_actual == prediction

    