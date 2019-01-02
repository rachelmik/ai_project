import numpy as np
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree

from DB_API import double_layer_params, pick_needed_features, get_set

default_alpha_linear = 0.04
default_gamma_poly = 1e-5
default_gamma_gaus = 1e-6


def get_linear_fit(X, y, alpha_array):
    linears = [Lasso(alpha=a) for a in alpha_array]
    linears = [linear.fit(X=X, y=y) for linear in linears]
    predicts = [linear.predict(X) for linear in linears]
    m = [mean_squared_error(p, y) for p in predicts]
    print("training linear ", m)
    return linears


def get_linear_predict(linears, X, y):
    predicts = [linear.predict(X) for linear in linears]
    m = [mean_squared_error(p, y) for p in predicts]
    print("test linear", m)
    return predicts


def single_linear_learn(alpha_array, is_shuffled=False):
    X, y = double_layer_params("single", "training", is_shuffled)
    linears = get_linear_fit(X, y, alpha_array)
    X, y = double_layer_params("single", "validation", is_shuffled)
    get_linear_predict(linears, X, y)


def double_linear_learn():
    X, usa_gross, rating = get_set("training")
    linear = Lasso(alpha=default_alpha_linear, normalize=True)
    multi = MultiOutputRegressor(linear)
    multi.fit(X, list(zip(usa_gross, rating)))
    predicts = multi.predict(X)
    print("training: ", mean_squared_error(predicts[:, 0], usa_gross))
    X, usa_gross, rating = get_set("validation")
    predicts = multi.predict(X)
    print("test: ", mean_squared_error(predicts[:, 0], usa_gross))


def get_svm_fit(X, y, gamma_array, kernel='rbf'):
    gps = [svm.SVR(kernel=kernel, degree=2, gamma=g) for g in gamma_array]
    gps = [gp.fit(X=X, y=y) for gp in gps]
    y_pred = [gp.predict(X) for gp in gps]
    m = [mean_squared_error(p, y) for p in y_pred]
    print("training results:", m)
    return gps


def get_svm_predict(gps, X, y):
    y_pred = [gp.predict(X) for gp in gps]
    m = [mean_squared_error(p, y) for p in y_pred]
    print("test results:", m)
    return y_pred


def svm_learn(gamma_array, kernel="rdf"):
    X, usa_gross = double_layer_params("single", "training")
    linear = get_linear_fit(X, usa_gross, [default_alpha_linear])[0]
    X_picked = pick_needed_features(linear, X)
    gps = get_svm_fit(X_picked, usa_gross, gamma_array, kernel)
    X, usa_gross = double_layer_params("single", "validation")
    X_picked = pick_needed_features(linear, X)
    get_svm_predict(gps, X_picked, usa_gross)


def multi_learn(first_layer="linear", second_layer="linear"):
    X_training, ratings_training = double_layer_params("rating", "training")
    X_test, ratings_test = double_layer_params("rating", "validation")
    if first_layer == "linear":
        linear_ratings = get_linear_fit(X_training, ratings_training, [default_alpha_linear])
        ratings_test = get_linear_predict(linear_ratings, X_test, ratings_test)[0]
    else:
        poly_ratings = get_svm_fit(X_training, ratings_training, [default_gamma_poly], kernel='poly')
        ratings_test = get_svm_predict(poly_ratings, X_test, ratings_test)[0]

    X_training, y_training = double_layer_params("single", "training")
    for i in range(len(X_training)):
        X_training[i].append(ratings_training[i])

    X_test, y_test = double_layer_params("single", "validation")
    for i in range(len(X_test)):
        X_test[i].append(ratings_test[i])

    if second_layer == "linear":
        linears_gross = get_linear_fit(X_training, y_training, [default_alpha_linear])
        get_linear_predict(linears_gross, X_test, y_test)
    else:
        poly_gross = get_svm_fit(X_training, y_training, [default_gamma_poly], kernel='poly')
        get_svm_predict(poly_gross, X_test, y_test)


def neural_network(num_of_layers, is_multi=False):
    print("getting training data")
    X, usa_gross, _ = get_set("training")
    linear = get_linear_fit(X, usa_gross, [default_alpha_linear])[0]
    training_res = []
    test_res = []
    for _ in range(5):
        X, usa_gross, rating = get_set("training")
        X_picked = pick_needed_features(linear, X)
        # X_picked = X
        net = MLPRegressor(hidden_layer_sizes=(100, )*num_of_layers)
        net = MultiOutputRegressor(net)
        if is_multi:
            net.fit(X_picked, list(zip(usa_gross, rating)))
        else:
            net.fit(X_picked, list(zip(usa_gross)))
        predicts = net.predict(X_picked)
        training_res.append(mean_squared_error(predicts[:, 0], usa_gross))
        X, usa_gross, rating = get_set("validation")
        X_picked = pick_needed_features(linear, X)
        # X_picked = X
        predicts = net.predict(X_picked)
        test_res.append(mean_squared_error(predicts[:, 0], usa_gross))
    return np.mean(training_res), np.std(training_res), np.mean(test_res), np.std(test_res)


def KNN_learn():
    X, usa_gross, _ = get_set("training")
    knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
    print("learning")
    knn.fit(X, usa_gross)
    X, usa_gross, _ = get_set("validation")
    predicts = knn.predict(X)
    print(mean_squared_error(predicts, usa_gross))


def get_binary_gross(gross):
    return 1 if gross > 200 else 0


def tree_classifier():
    clf = tree.DecisionTreeClassifier()
    X, usa_gross, _ = get_set("training")
    y = [get_binary_gross(i) for i in usa_gross]
    clf.fit(X, y)
    X_small = [X[i] for i in range(len(y)) if y[i] == 0]
    y_small = [usa_gross[i] for i in range(len(y)) if y[i] == 0]
    X_big = [X[i] for i in range(len(y)) if y[i] == 1]
    y_big = [usa_gross[i] for i in range(len(y)) if y[i] == 1]
    print("Big fit:")
    linear_big = get_linear_fit(X_big, y_big, [default_alpha_linear])
    print("Small fit:")
    linear_small = get_linear_fit(X_small, y_small, [default_alpha_linear])
    X, usa_gross, _ = get_set("validation")
    y = np.array([get_binary_gross(i) for i in usa_gross])
    p = clf.predict(X)

    X_small = [X[i] for i in range(len(p)) if p[i] == 0]
    y_small = [usa_gross[i] for i in range(len(p)) if p[i] == 0]
    X_big = [X[i] for i in range(len(p)) if p[i] == 1]
    y_big = [usa_gross[i] for i in range(len(p)) if p[i] == 1]
    print("Big predict:")
    get_linear_predict(linear_big, X_big, y_big)
    print("Small predict:")
    get_linear_predict(linear_small, X_small, y_small)


def net_layers(layers_array, is_multi=False):
    training = []
    test = []
    training_std = []
    test_std = []
    for i in layers_array:
        print(i)
        r1, r2, r3, r4 = neural_network(i, is_multi)
        training.append(r1)
        training_std.append(r2)
        test.append(r3)
        test_std.append(r4)
    print(training)
    print(training_std)
    print(test)
    print(test_std)


def new_prediction(X):
    X_training, ratings_training = double_layer_params("rating", "training")
    X_val, ratings_val = double_layer_params("rating", "validation")
    X_training += X_val
    ratings_training += ratings_val
    # poly_ratings = get_svm_fit(X_training, ratings_training, [default_gamma_poly], kernel='poly')
    # ratings_test = get_svm_predict(poly_ratings, X, [0]*len(X))[0]
    linear_rating = get_linear_fit(X_training, ratings_training, [default_alpha_linear])
    ratings_test = get_linear_predict(linear_rating, X, [0]*len(X))[0]

    X_training, y_training = double_layer_params("single", "training")
    for i in range(len(X_training)):
        X_training[i].append(ratings_training[i])

    for i in range(len(X)):
        X[i].append(ratings_test[i])

    linears_gross = get_linear_fit(X_training, y_training, [default_alpha_linear])
    return get_linear_predict(linears_gross, X, [0]*len(X))

