import json
import os
import time
from random import shuffle

import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from path_to_data import get_data_path

all_genres_list = [' Thriller', ' Mystery', ' Biography', ' Music', 'War', ' Horror', ' Drama', ' Crime', ' History',
                   ' Comedy', ' Family', ' Animation', ' Musical', ' Sport', ' News', ' Romance', ' Fantasy', ' Action',
                   ' Sci-Fi', ' Western', ' Adventure', ' War']

path_to_data = get_data_path()


def read_json_file(year, json_file):
    with open(path_to_data + str(year) + '/' + json_file) as f:
        movie = json.load(f)
    return movie


def get_data(min_year, max_year, is_shuffled=False):
    if is_shuffled:
        return get_data_with_actors_shuffles(min_year, max_year)
    db = []
    for year in range(min_year, max_year):
        print(year)
        json_list = os.listdir(path_to_data + str(year))
        for json_file in json_list:
            if json_file == "desktop.ini":
                continue
            with open(path_to_data + str(year) + '/' + json_file) as f:
                movie = json.load(f)
                genres = [0] * 22
                for g in movie['genres']:
                    index = all_genres_list.index(g)
                    genres[index] = 1
                movie['genres_array'] = genres
                db.append(movie)
    return db


def get_data_with_actors_shuffles(min_year, max_year):
    db = []
    for year in range(min_year, max_year):
        print(year)
        json_list = os.listdir(path_to_data + str(year))
        for json_file in json_list:
            with open(path_to_data + str(year) + '/' + json_file) as f:
                movie = json.load(f)
                genres = [0] * 22
                for g in movie['genres']:
                    index = all_genres_list.index(g)
                    genres[index] = 1
                movie['genres_array'] = genres
                shuffled_actors_array = get_shuffled_actors_array(movie["cast_enriched"])
                for array in shuffled_actors_array:
                    movie["cast_enriched"] = array
                    db.append(movie)
    return db


def check_len(data, func):
    if len(data) > 0:
        return func(data)
    return 0


def get_person_gross(person, max_year):
    movies = person["movies"]
    movies = [m for m in movies if m.get("year") < max_year]
    movies = sorted(movies, key=lambda k: k['year'], reverse=True)
    gross = []
    complex_gross = []
    places = []
    years = []
    for m in movies:
        places.append(m.get("place"))
        years.append(m.get("year"))
        gross_usa = m.get("details").get("Gross USA")
        if gross_usa is not None and isinstance(gross_usa, int):
            movie_gross = m.get("details").get("Gross USA")
        else:
            movie_gross = m.get("details").get("Cumulative Worldwide Gross")
        if movie_gross is not None and isinstance(gross_usa, int):
            movie_gross = movie_gross / 1e6
            gross.append(movie_gross)
            complex_gross.append(movie_gross / m.get("place"))

    return gross, complex_gross, places, years


# getting personal data from perople files to movie files
def enrich_person(person, max_year):
    gross, complex_gross, places, years = get_person_gross(person, max_year)
    gross = gross[:5]
    complex_gross = complex_gross[:5]
    num_of_movies = len(gross)
    person["average_gross"] = check_len(gross, np.mean)
    person["max_gross"] = max(gross) if num_of_movies != 0 else 0
    person["std_gross"] = check_len(gross, np.std)
    if num_of_movies > 1:
        gradient = list(np.gradient(gross))
        person["gradient"] = gradient
        person["avg_gradient"] = check_len(gradient, np.mean)
    person["average_complex_gross"] = check_len(complex_gross, np.mean)
    return person


def get_file_name(person, files_dict):
    url = person["url"]
    options = ["/".join(url.split("/")[:-1]) + "/?ref_=ttfc_fc_t1", "_".join(url.split("_")[:-1]) + "_t1",
               "/".join(url.split("/")[:-1]) + "/?ref_=ttfc_fc_cl_t1"]
    for o in options:
        try:
            f = files_dict[o]
            return f
        except KeyError:
            continue
    return None


def get_movie_person_data(movie):
    with open(path_to_data + "people_links_dict.json") as f:
        files_dict = json.load(f)

    stuff_files = {}
    for person_type in ["producer", "director", "writer"]:
        if len(movie[person_type + "s"]) > 0:
            stuff_files[person_type] = get_file_name(movie[person_type + "s"][0], files_dict[person_type])
    cast_files = [get_file_name(i, files_dict["actor"]) for i in movie["cast"]]
    year = movie["year"]

    stuff = {}
    for p_type, file_name in stuff_files.items():
        with open("{}{}/{}".format(path_to_data, p_type, file_name)) as f:
            person = json.load(f)
            stuff[p_type] = enrich_person(person, year)

    cast = []
    for actor_file_name in cast_files:
        if actor_file_name is None:
            continue
        with open("{}actor/{}".format(path_to_data, actor_file_name)) as f:
            person = json.load(f)
            cast.append(enrich_person(person, year))

    actresses = [i for i in cast if i["type"] == "actress"]
    movie["actress_ratio"] = len(actresses) / len(cast) if len(cast) != 0 else 0
    ages = [i["age"] for i in cast if i["age"] != 0]
    movie["average_age"] = check_len(ages, np.mean)
    for k, v in stuff.items():
        movie["{}_enriched".format(k)] = v
    movie["cast_enriched"] = cast

    return movie


# getting features from movie files
def get_list_of_details_feature(db, feature):
    return [movie.get('details').get(feature) for movie in db]


def get_list_of_feature(db, feature):
    return [movie.get(feature) for movie in db]


def remove_nones(list1, list2, list3=None):
    if list3:
        try:
            return zip(*[(i, j, k) for i, j, k in zip(list1, list2, list3) if None not in i and j != 0])
        except TypeError:
            return zip(*[(i, j, k) for i, j, k in zip(list1, list2, list3) if i is not None and j != 0])
    try:
        return zip(*[(i, j) for i, j in zip(list1, list2) if None not in i and j != 0])
    except TypeError:
        return zip(*[(i, j) for i, j in zip(list1, list2) if i is not None and j != 0])


def get_date_ts(date):
    year = time.strptime(str(date.tm_year), '%Y')
    return time.mktime(date) - time.mktime(year)


def parse_date(orig_date):
    date = None
    if orig_date is None:
        return None
    for form in ['%d %B %Y ', '%B %Y ']:
        try:
            date = get_date_ts(time.strptime(orig_date.split("(")[0], form))
            date = int(date / (24 * 60 * 60)) + 1
            break
        except ValueError:
            continue

    return date


def get_gross(db):
    for movie in db:
        all_gross = movie.get('details').get("Cumulative Worldwide Gross")
        if movie.get('details').get("Gross USA") is None:
            movie["details"]["Gross USA"] = all_gross
    gross = get_list_of_details_feature(db, "Gross USA")
    return [i / 10 ** 6 for i in gross]


def get_ratings(db):
    ratings = []
    for movie in db:
        if movie.get('rating') is not None:
            ratings.append(movie.get('rating'))
    return ratings


def get_genre_dict(db):
    genres = get_list_of_feature(db, "genres")
    usa_gross = get_gross(db)
    genres, usa_gross = remove_nones(genres, usa_gross)
    all_genres = set([item for sublist in genres for item in sublist])
    genre_dict = {i: [] for i in all_genres}
    for genre, gross in zip(genres, usa_gross):
        for k, v in genre_dict.items():
            if k in genre:
                v.append(gross)
    return genre_dict


def get_genre(movie, genre_dict):
    genres = {k: np.mean(v) for k, v in genre_dict.items() if len(v) > 0}
    movie_genres = movie.get("genres")
    genre_gross = [genres[i] for i in movie_genres if i in genres.keys()]
    avg_movies_gross_by_genre = 0
    max_movies_gross_by_genre = 0
    if len(genre_gross) != 0:
        avg_movies_gross_by_genre = np.mean(genre_gross)
        max_movies_gross_by_genre = max(genre_gross)
    return avg_movies_gross_by_genre, max_movies_gross_by_genre
    # movie_genres = get_list_of_feature(db, "genres")
    # avg_movies_gross_by_genre = []
    # max_movies_gross_by_genre = []
    # for movie in movie_genres:
    #     genre_gross = [genres[i] for i in movie if i in genres.keys()]
    #     if len(genre_gross) == 0:
    #         avg_movies_gross_by_genre.append(0)
    #         max_movies_gross_by_genre.append(0)
    #     else:
    #         avg_movies_gross_by_genre.append(sum(genre_gross) / len(genre_gross))
    #         max_movies_gross_by_genre.append(max(genre_gross))
    #
    # return avg_movies_gross_by_genre, max_movies_gross_by_genre


def append_none(data, func):
    result = []
    for i in data:
        try:
            result.append(func(i))
        except Exception:
            result.append(None)

    return result


# def get_person_params(person_list):
#     features = ["average_gross", "max_gross", "std_gross", "avg_gradient", "average_complex_gross"]
#     params = []
#     for feature in features:
#         # person_features = []
#         # for person in person_list:
#         #     if person is None:
#         #         person_features.append(None)
#         #     else:
#         #         person_features.append(person.get(feature))
#         person_features = append_none(person_list, lambda person: person.get(feature))
#         params.append(person_features)
#     return params

def pad(list_to_pad, pad_with, length):
    return list_to_pad + [pad_with] * (length - len(list_to_pad))


def get_person_params(person, max_year):
    num_of_movies = 3
    default = [0] * num_of_movies
    if person is None:
        return default
    gross, complex_gross, places, years = get_person_gross(person, max_year)
    if len(years) == 0:
        return default
    if len(years) > num_of_movies:
        year_diff = years[0] - years[num_of_movies - 1]
    else:
        year_diff = years[0] - years[-1]
    return pad(gross[:num_of_movies], 0, num_of_movies)


def get_shuffled_actors_array(actors):
    shuffled_actors_array = []
    shuffled_actors_array.append(actors.copy())
    for i in range(4):
        if None in actors:
            shuffled_actors_array.append(actors.copy())
            continue
        shuffle(actors)
        shuffled_actors_array.append(actors.copy())
    return shuffled_actors_array


def get_movie_params(movie):
    date = parse_date(movie.get('details').get("Release Date"))
    num_of_actors = len(movie.get("cast"))
    runtime = movie.get('details').get("Runtime")
    woman_ratio = movie.get("actress_ratio")
    avg_cast_age = movie.get("average_age")
    params = [date, num_of_actors, runtime, woman_ratio, avg_cast_age]
    params += movie.get("genres_array")
    producer = movie.get("producer_enriched")
    director = movie.get("director_enriched")
    writer = movie.get("writer_enriched")
    actors = movie.get("cast_enriched")
    num_of_actors_picked = 35
    people = [producer, director, writer] + pad(actors[:num_of_actors_picked], None, num_of_actors_picked)
    year = movie.get("year")
    for p in people:
        params += get_person_params(p, year)

    return params


def get_all_params(db):
    params = [get_movie_params(m) for m in db]
    usa_gross = get_gross(db)
    ratings = get_ratings(db)
    return usa_gross, params, ratings


# creating histograms
def create_hist(values_dic, is_big):
    hist = {k: sum(v) / len(v) for k, v in values_dic.items() if len(v) > 0}
    hist = sorted(hist.items(), key=lambda x: x[0])
    fig, ax = plt.subplots()
    index = np.arange(len(hist))
    rects1 = ax.bar(index, [i[1] for i in hist])
    if is_big:
        ax.set_xticks(np.arange(9) * 5 - 0.5)
        ax.set_xticklabels([i for i in range(0, 450, 50)])
    else:
        ax.set_xticks(np.arange(len(hist)))
        ax.set_xticklabels([i[0] for i in hist])
    plt.show()


def create_range_hist(usa_gross, param):
    param, usa_gross = remove_nones(param, usa_gross)
    n, bins, patches = plt.hist(param, 50, weights=usa_gross)
    hist = {b: [] for b in bins[:-1]}
    for i in range(50):
        hist[bins[i]] = [v for c, v in zip(param, usa_gross) if bins[i] <= c < bins[i + 1]]
    create_hist(hist, True)


def create_bars(usa_gross, param):
    param, usa_gross = remove_nones(param, usa_gross)
    hist = {b: [] for b in set(param)}
    for p, g in zip(param, usa_gross):
        hist[p].append(g)
    create_hist(hist, False)


def get_set(set_type, is_shuffled=False):
    shuffled = "_shuffled" if is_shuffled else ""
    set_file_path = path_to_data + "{}_set{}.json".format(set_type, shuffled)
    if os.path.isfile(set_file_path):
        with open(set_file_path) as f:
            X, usa_gross, ratings = json.load(f)
    else:
        if set_type == "training":
            db = get_data(2007, 2015, is_shuffled)
        elif set_type == "test":
            db = get_data(2015, 2017, is_shuffled)
        else:
            raise ValueError("{} not a valid set type".format(set_type))
        usa_gross, X, ratings = get_all_params(db)
        with open(set_file_path, "w+") as f:
            json.dump((X, usa_gross, ratings), f)

    X, usa_gross, ratings = remove_nones(X, usa_gross, ratings)
    return X, usa_gross, ratings


def get_diff(predicts, gross):
    diff = []
    i = 0
    for p in predicts:
        d = abs(gross[i] - p)
        diff.append(d)
        i += 1
    return diff


def print_diff_partition(diff):
    d_range = [0]*11
    for d in diff:
        d_range[int(d) % 10] += 1
    for i in range(11):
        l = i * 10
        r = (i+1) * 10
        if r > 100:
            r = 500
        p = float("{0:.2f}".format(d_range[i]/len(diff)*100))

        print('movies with diff between {} to {} millions : {}%'.format(l, r, p))


def create_all_histograms():
    # db = get_data(2007, 2017)
    # release_date = get_list_of_details_feature(db, "Release Date")
    # dates = [parse_date(d) for d in release_date]
    # cast = get_list_of_feature(db, "cast")
    # cast = [len(i) for i in cast]
    # year = get_list_of_feature(db, "year")
    # runtime = get_list_of_details_feature(db, "Runtime")
    # usa_gross = get_gross(db)
    with open(f'{path_to_data}histogram_data.json') as f:
        data = json.load(f)

    usa_gross = data["gross"]

    create_bars(usa_gross, data["year"])
    # create_range_hist(usa_gross, data["dates"])
    # create_range_hist(usa_gross, data["cast"])
    # create_range_hist(usa_gross, data["runtime"])
    # genres = get_genre_dict(db)
    # create_hist(data["genres"], False)
    # with open(f'{path_to_data}histogram_data.json', "w+") as f:
    #     json.dump({"dates": dates, "cast": cast, "year": year, "runtime": runtime,
    #                "genres": genres, "gross": usa_gross}, f)
    #


def pick_needed_features(linear, X):
    needed_feature_indexes = [i for i in range(len(linear.coef_)) if linear.coef_[i]]
    X = np.array(X)
    picked_X = [X[:, i] for i in needed_feature_indexes]
    return np.array(list(zip(*picked_X)))


def double_layer_params(mode, set_type, is_shuffled=False):
    X, usa_gross, rating = get_set(set_type, is_shuffled)
    if mode == "rating":
        y = rating
    elif mode == "single":
        y = usa_gross
    else:
        raise RuntimeError

    return X, y


# linear learning
def get_linear_fit(X, y):
    linears = [Lasso(alpha=a) for a in [80]]
    print("getting training data")
    linears = [linear.fit(X=X, y=y) for linear in linears]
    predicts = [linear.predict(X) for linear in linears]
    m = [mean_squared_error(p, y) for p in predicts]
    print("training linear ", m)
    return linears


def get_linear_predict(linears, X, y):
    print("getting test data")
    predicts = [linear.predict(X) for linear in linears]
    # diff = get_diff(predicts, usa_gross)
    # print_diff_partition(diff)
    # plt.plot(diff)
    # plt.show()
    m = [mean_squared_error(p, y) for p in predicts]
    print("test linear", m)
    return predicts


def single_linear_learn(is_shuffled=False):
    X, y = double_layer_params("single", "training", is_shuffled)
    linears = get_linear_fit(X, y)
    X, y = double_layer_params("single", "test", is_shuffled)
    get_linear_predict(linears, X, y)


def double_linear_learn():
    X, y = double_layer_params("rating", "training")
    linears_raiting = get_linear_fit(X, y)
    ratings_training = get_linear_predict(linears_raiting, X, y)
    X, y = double_layer_params("rating", "test")
    ratings_test = get_linear_predict(linears_raiting, X, y)

    X, y = double_layer_params("single", "training")
    for i in range(len(X)):
        X[i].append(ratings_training[i])
    linears_gross = get_linear_fit(X, y)

    X, y = double_layer_params("single", "test")
    for i in range(len(X)):
        X[i].append(ratings_test[i])
    get_linear_predict(linears_gross, X, y)


def gaussian_learn(is_shuffled=False):
    linear = get_linear_fit(is_shuffled)[0]
    print("getting training data")
    X, usa_gross, _ = get_set("training", is_shuffled)
    X_picked = pick_needed_features(linear, X)
    gps = [svm.SVR(kernel='poly', degree=2) for g in [1]]
    print("learning")
    gps = [gp.fit(X=X_picked, y=usa_gross) for gp in gps]
    print("predicting on training")
    y_pred = [gp.predict(X_picked) for gp in gps]
    m = [mean_squared_error(p, usa_gross) for p in y_pred]
    print("training results:", m)
    print("getting test data")
    X, usa_gross, _ = get_set("test", is_shuffled)
    X_picked = pick_needed_features(linear, X)
    y_pred = [gp.predict(X_picked) for gp in gps]
    # diff = get_diff(y_pred, usa_gross)
    # print_diff_partition(diff)
    # plt.figure()
    # plt.plot(diff)
    # plt.show()
    m = [mean_squared_error(p, usa_gross) for p in y_pred]
    print("test results:", m)
    # plt.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
    # plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
    # plt.plot(x, y_pred, 'b-', label=u'Prediction')
    # plt.fill(np.concatenate([x, x[::-1]]),
    #          np.concatenate([y_pred - 1.9600 * sigma,
    #                          (y_pred + 1.9600 * sigma)[::-1]]),
    #          alpha=.5, fc='b', ec='None', label='95% confidence interval')
    # plt.xlabel('$x$')
    # plt.ylabel('$f(x)$')
    # plt.ylim(-10, 20)
    # plt.legend(loc='upper left')


def neural_network():
    print("getting training data")
    X, usa_gross, _ = get_set("training")
    # linear = get_linear_fit()[0]
    # X_picked = pick_needed_features(linear, X)
    X_picked = X
    net = MLPRegressor(hidden_layer_sizes=(100, 100, 100, 100, 100, 100))
    net.fit(X_picked, usa_gross)
    predicts = net.predict(X_picked)
    print("training results:",  mean_squared_error(predicts, usa_gross))
    X, usa_gross, _ = get_set("test")
    X_picked = X
    # X_picked = pick_needed_features(linear, X)
    predicts = net.predict(X_picked)
    print("test results:", mean_squared_error(predicts, usa_gross))


single_linear_learn()

# gaussian_learn()

# neural_network()

# create_all_histograms()
# x = [10**(-i) for i in range(10)]
# training = [4706.0844146979225, 4705.538544429718, 4701.5240069889505, 4649.462839836307, 4463.845319768954, 3753.198360360228, 3396.6738359355327, 4349.101691108962, 4728.000610503838, 4767.971936639001]
# test = [5767.125207842541, 5766.5398678762285, 5761.986805557549, 5707.238276485438, 5518.578684352951, 4753.367065523703, 4223.027292326871, 5273.255382710276, 5718.06296263596, 5766.08767727686]
# plt.semilogx(x, training, label="training set")
# plt.semilogx(x, test, label="test set")
# plt.grid()
# plt.xlabel("Gamma")
# plt.ylabel("MSE (in millions USD)")
# plt.title("SVM with Gaussian kernel - results vs gamma")
# plt.legend()
# plt.show()