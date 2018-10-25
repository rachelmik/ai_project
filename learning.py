import numpy as np
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor

from DB_API import double_layer_params, pick_needed_features, get_set

default_alpha_linear = 0.04
default_gamma_poly = 1e-5
default_gamma_gaus = 1e-6


def get_linear_fit(X, y, alpha_array):
    linears = [Lasso(alpha=a, normalize=True) for a in alpha_array]
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


def single_linear_learn(alpah_array, is_shuffled=False):
    X, y = double_layer_params("single", "training", is_shuffled)
    linears = get_linear_fit(X, y, alpah_array)
    X, y = double_layer_params("single", "test", is_shuffled)
    get_linear_predict(linears, X, y)


def double_linear_learn():
    X, usa_gross, rating = get_set("training")
    linear = Lasso(alpha=default_alpha_linear, normalize=True)
    multi = MultiOutputRegressor(linear)
    multi.fit(X, list(zip(usa_gross, rating)))
    predicts = multi.predict(X)
    print("training: ", mean_squared_error(predicts[:, 0], usa_gross))
    X, usa_gross, rating = get_set("test")
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
    print("training results:", m)
    return y_pred


def svm_learn(gamma_array, kernel="rdf"):
    X, usa_gross = double_layer_params("single", "training")
    linear = get_linear_fit(X, usa_gross, [default_alpha_linear])[0]
    X_picked = pick_needed_features(linear, X)
    gps = get_svm_fit(X_picked, usa_gross, gamma_array, kernel)
    X, usa_gross = double_layer_params("single", "test")
    X_picked = pick_needed_features(linear, X)
    get_svm_predict(gps, X_picked, usa_gross)


def multi_learn(first_layer="linear", second_layer="linear"):
    X_training, ratings_training = double_layer_params("rating", "training")
    X_test, ratings_test = double_layer_params("rating", "test")
    if first_layer == "linear":
        linear_ratings = get_linear_fit(X_training, ratings_training, [default_alpha_linear])
        ratings_test = get_linear_predict(linear_ratings, X_test, ratings_test)[0]
    else:
        poly_ratings = get_svm_fit(X_training, ratings_training, [default_gamma_poly], kernel='poly')
        ratings_test = get_svm_predict(poly_ratings, X_test, ratings_test)[0]

    X_training, y_training = double_layer_params("single", "training")
    for i in range(len(X_training)):
        X_training[i].append(ratings_training[i])

    X_test, y_test = double_layer_params("single", "test")
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
        net = MLPRegressor(hidden_layer_sizes=(100, )*num_of_layers)
        net = MultiOutputRegressor(net)
        if is_multi:
            net.fit(X_picked, list(zip(usa_gross, rating)))
        else:
            net.fit(X_picked, list(zip(usa_gross)))
        predicts = net.predict(X_picked)
        training_res.append(mean_squared_error(predicts[:, 0], usa_gross))
        X, usa_gross, rating = get_set("test")
        X_picked = pick_needed_features(linear, X)
        predicts = net.predict(X_picked)
        test_res.append(mean_squared_error(predicts[:, 0], usa_gross))
    return np.mean(training_res), np.std(training_res), np.mean(test_res), np.std(test_res)


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

