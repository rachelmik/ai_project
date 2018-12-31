import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree

from DB_API import double_layer_params, pick_needed_features, get_set, get_data, get_all_params

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
    print("test results:", m)
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
        # X_picked = X
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
        # X_picked = X
        predicts = net.predict(X_picked)
        test_res.append(mean_squared_error(predicts[:, 0], usa_gross))
    return np.mean(training_res), np.std(training_res), np.mean(test_res), np.std(test_res)


def KNN_learn():
    X, usa_gross, _ = get_set("training")
    knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
    print("learning")
    knn.fit(X, usa_gross)
    X, usa_gross, _ = get_set("test")
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
    X, usa_gross, _ = get_set("test")
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
    poly_ratings = get_svm_fit(X_training, ratings_training, [default_gamma_poly], kernel='poly')
    ratings_test = get_svm_predict(poly_ratings, X, [0]*len(X))[0]

    X_training, y_training = double_layer_params("single", "training")
    for i in range(len(X_training)):
        X_training[i].append(ratings_training[i])

    for i in range(len(X)):
        X[i].append(ratings_test[i])

    linears_gross = get_linear_fit(X_training, y_training, [default_alpha_linear])
    return get_linear_predict(linears_gross, X, [0]*len(X))


# single_linear_learn([0.1,0.5,1,5,10, 20, 50, 100,200])

# X, usa_gross, _ = get_set("test_fin")
# predict = new_prediction(X)
# print(mean_squared_error(usa_gross, predict[0]))

# net_layers([1, 2, 3, 4, 5, 6])
# train_no = [135.61257399733705, 144.18723477894042, 121.26157033333975, 105.38511569703894, 129.373735457198, 184.03270375374132]
# train_std_no = [23.31613327478266, 32.14218660837528, 31.48340533365122, 15.932101859452125, 35.15642329816026, 46.00374807676252]
# test_no = [3172.836127257632, 3015.558887629304, 2819.609853961562, 2686.5805548307144, 2469.7670836909556, 2579.8499843068935]
# test_std_no = [227.1145303442531, 209.9011036037836, 281.14811989194794, 173.93155999946592, 125.8331763263739, 203.42771480617097]
# train = [185.6662336310445, 191.02359253165167, 177.3343843504012, 182.71063859967472, 221.17032001913026, 153.2504159322522]
# train_std = [24.971536049196626, 21.58817220828703, 24.51345583146729, 66.67966498190431, 20.18517238854704, 22.398346771087418]
# test = [3442.414356559424, 3010.0554065587526, 2885.3803873549045, 2744.2825934090424, 2518.3444625422744, 2430.3294854884475]
# test_std = [365.8059943230088, 135.9058637454946, 218.1087015781143, 304.73890744538653, 107.54065183651758, 110.75337149461653]
#
# x = [i for i in range(1,7)]
x = [0.1, 0.5, 1, 5, 10, 20, 50, 100, 200]
train = [1118.2653891704176, 1133.616963425744, 1153.5797451593473, 1209.5632562866183, 1214.8448437637571, 1228.6949560058854, 1263.5740893419324, 1305.7281414340669, 1412.7969881678964]
test = [1377.6900651260246, 1385.3362722851982, 1390.22293012402, 1415.0242236290383, 1395.1904391566095, 1356.5228220946356, 1323.1210142466987, 1336.741105678674, 1433.752309400869]
#
plt.plot(x, train, label="training set - picked")
plt.plot(x, test, label="test set - picked")
# plt.plot(x, train_std_no, label="training set - not picked", linestyle="--")
# plt.plot(x, test_std_no, label="test set - not picked",  linestyle="--")
plt.xlabel("Number of layers")
plt.ylabel("MSE (in millions USD")
plt.grid()
plt.legend()
plt.title("Neural network - Number of layers vs std of results")
plt.show()
