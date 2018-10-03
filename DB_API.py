import json
import os
import time
from random import shuffle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from path_to_data import get_data_path

all_genres_list = [' Thriller', ' Mystery', ' Biography', ' Music', 'War', ' Horror', ' Drama', ' Crime', ' History',
                   ' Comedy', ' Family', ' Animation', ' Musical', ' Sport', ' News', ' Romance', ' Fantasy', ' Action',
                   ' Sci-Fi', ' Western', ' Adventure', ' War']

path_to_data = get_data_path()


def read_json_file(year, json_file):
    with open(path_to_data + str(year) + '/' + json_file) as f:
        movie = json.load(f)
    return movie


def get_data(min_year, max_year):
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


def remove_nones(list1, list2):
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
    default = [0] * 7
    if person is None:
        return default
    gross, complex_gross, places, years = get_person_gross(person, max_year)
    num_of_movies = 5
    if len(years) == 0:
        return default
    if len(years) > num_of_movies:
        year_diff = years[0] - years[num_of_movies - 1]
    else:
        year_diff = years[0] - years[-1]
    return pad(gross[:num_of_movies], 0, num_of_movies) + [np.mean(places[:num_of_movies]), year_diff]


def get_shuffled_actors_array(actors):
    shuffled_actors_array= []
    shuffled_actors_array.append(actors.copy())
    for i in range (4):
        if None in actors:
            shuffled_actors_array.append(actors.copy())
            continue
        shuffle(actors)
        shuffled_actors_array.append(actors.copy())
    # shuffled_actors = [actors]
    # temp_actors = actors
    # for i in range(1, 15):
    #     try:
    #         last = temp_actors.pop(0)
    #         temp_actors.append(last)
    #         shuffled_actors.append(temp_actors)
    #     except IndexError:
    #         break;
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
    people = [producer, director, writer] + pad(actors[:15], None, 15)
    year = movie.get("year")
    for p in people:
        params += get_person_params(p, year)

    return params


def get_all_params(db):
    # release_date = get_list_of_details_feature(db, "Release Date")
    # dates = [parse_date(date) for date in release_date]
    # cast = get_list_of_feature(db, "cast")
    # cast_num = [len(i) for i in cast]
    # runtime = get_list_of_details_feature(db, "Runtime")
    # usa_gross = get_gross(db)
    # avg_movies_gross_by_genre, max_movies_gross_by_genre = [get_genre(movie, genres) for movie in db]
    # avg_cast_age = get_list_of_feature(db, "average_age")
    # women_ratio = get_list_of_feature(db, "actress_ratio")
    # params = [cast_num, avg_movies_gross_by_genre, max_movies_gross_by_genre, women_ratio, avg_cast_age]
    # params += get_person_params(get_list_of_feature(db, "producer_enriched"))
    # params += get_person_params(get_list_of_feature(db, "director_enriched"))
    # params += get_person_params(get_list_of_feature(db, "writer_enriched"))
    # movies_cast = get_list_of_feature(db, "cast_enriched")
    # for i in range(15):
    #     actor_by_index = append_none(movies_cast, lambda actor: actor[i])
    #     params += get_person_params(actor_by_index)
    params = [get_movie_params(m) for m in db]
    usa_gross = get_gross(db)
    return usa_gross, params


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


def create_range_hist(db, param):
    usa_gross = get_gross(db)
    param, usa_gross = remove_nones(param, usa_gross)
    n, bins, patches = plt.hist(param, 50, weights=usa_gross)
    hist = {b: [] for b in bins[:-1]}
    for i in range(50):
        hist[bins[i]] = [v for c, v in zip(param, usa_gross) if bins[i] <= c < bins[i + 1]]
    create_hist(hist, True)


def create_bars(db, param):
    usa_gross = get_gross(db)
    param, usa_gross = remove_nones(param, usa_gross)
    hist = {b: [] for b in set(param)}
    for p, g in zip(param, usa_gross):
        hist[p].append(g)
    create_hist(hist, False)


# linear learning
def get_linear_fit(db):
    usa_gross, X = get_all_params(db)
    linear = Lasso(alpha=1)
    X, usa_gross = remove_nones(X, usa_gross)
    return linear.fit(X=X, y=usa_gross)

def get_gaussian_fit(db):
    usa_gross, X = get_all_params(db)
    gp = GaussianProcessRegressor(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1e-1,
                         random_start=100)
    X, usa_gross = remove_nones(X, usa_gross)
    return gp.fit(X=X, y=usa_gross)

def get_gaussian_predict(db, gp):
    usa_gross, X = get_all_params(db)
    X, usa_gross = remove_nones(X, usa_gross)
    predicts = gp.predict(X)
    return mean_squared_error(predicts, usa_gross)


def get_linear_predict(db, linear):
    usa_gross, X = get_all_params(db)
    X, usa_gross = remove_nones(X, usa_gross)
    predicts = linear.predict(X)
    diff = get_diff(predicts, usa_gross)
    print_diff_partition(diff)
    plt.plot(diff)
    plt.show()
    return mean_squared_error(predicts, usa_gross)


def get_diff(predicts, gross):
    diff = []
    i = 0
    for p in predicts:
        d = abs(gross[i] - p)
        diff.append(d)
        i += 1
    return diff


def print_diff_partition(diff):
    range = [0]*11
    for d in diff:
        if d <10:
            range[0]+=1
        elif d<20:
            range[1]+=1
        elif d<30 :
            range[2]+=1
        elif d<40:
            range[3]+=1
        elif d<50:
            range[4]+=1
        elif d<60:
            range[5]+=1
        elif d<70:
            range[6]+=1
        elif d<80:
            range[7]+=1
        elif d<90:
            range[8]+=1
        elif d<100:
            range[9]+=1
        else:
            range[10]+=1
    i=0
    while i<11:
        l = (i)*10
        r = (i+1)*10
        if r > 100:
            r=500
        p =float("{0:.2f}".format(range[i]/len(diff)*100))

        print(f'movies with diff between {l} to {r} millions : {p}%')
        i+=1


def create_all_histograms():
    db = get_data(2007, 2017)
    release_date = get_list_of_details_feature(db, "Release Date")
    dates = [parse_date(d) for d in release_date]
    cast = get_list_of_feature(db, "cast")
    cast = [len(i) for i in cast]
    year = get_list_of_feature(db, "year")
    runtime = get_list_of_details_feature(db, "Runtime")

    create_bars(db, year)
    create_range_hist(db, dates)
    create_range_hist(db, cast)
    create_range_hist(db, runtime)
    genres = get_genre_dict(db)
    create_hist(genres, False)


def learn():
    # with open(path_to_data + "db_learn.json") as f:
    #     db = json.load(f)
    #db = get_data_with_actors_shuffles(2007, 2015)
    db = get_data(2007,2015)
    linear = get_linear_fit(db)
    # with open(path_to_data + "db_test.json") as f:
    #     db = json.load(f)
    #db = get_data_with_actors_shuffles(2015, 2017)
    db = get_data(2015, 2017)
    print(get_linear_predict(db, linear))




#todo: check if we are using the Gaussian currectly
#todo: fix gaussian learn with shuffled data
#todo: add L1,L2 regulation
def gaussian_learn():
    db = get_data(2007,2015)
    #db = get_data_with_actors_shuffles(2007,2015)
    usa_gross, X = get_all_params(db)
    X, usa_gross = remove_nones(X, usa_gross)
    #todo: maby choos another kernel that is not RBF
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp.fit(X=X, y=usa_gross)
    #db = get_data_with_actors_shuffles(2015, 2017)
    db = get_data(2015, 2017)
    usa_gross, X = get_all_params(db)
    X, usa_gross = remove_nones(X, usa_gross)
    # Mesh the input space for evaluations of the real function, the prediction and
    # its MSE
    #x = np.atleast_2d(np.linspace(0, 10, 1000)).T
    y_pred, sigma = gp.predict(X, return_std=True)
    # Plot the function, the prediction and the 95% confidence interval based on
    # the MSE
    diff = get_diff(y_pred, usa_gross)
    print_diff_partition(diff)
    plt.figure()
    plt.plot(diff)
    # plt.show()
    #todo: fix mean_squared_error func for gaussian
    return mean_squared_error(y_pred, usa_gross)
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


learn()


gaussian_learn()


# todo: gaos mesveg for all runs
# todo: get inside the results and to see th presentege of the errors - DIDNT WORK!
# todo: run with/without shuffle
# todo: write the rapport
# todo: add L2 regulation
