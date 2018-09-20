import json
import os
import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from path_to_data import get_data_path

all_genres_list = [' Thriller', ' Mystery', ' Biography', ' Music', 'War', ' Horror', ' Drama', ' Crime', ' History',
                   ' Comedy', ' Family', ' Animation', ' Musical', ' Sport', ' News', ' Romance', ' Fantasy', ' Action',
                   ' Sci-Fi', ' Western', ' Adventure', ' War']

path_to_data = get_data_path()


def get_data(min_year, max_year):
    db = []
    for year in range(min_year, max_year):
        print(year)
        json_list = os.listdir(path_to_data + str(year))
        for json_file in json_list:
            with open(path_to_data + str(year) + '/' + json_file) as f:
                movie = json.load(f)
                genres = [0]*22
                for g in movie['genres']:
                    index = all_genres_list.index(g)
                    genres[index] = 1
                movie['genres'] = genres
                db.append(movie)
    return db


def check_len(data, func):
    if len(data) > 0:
        return func(data)
    return 0


# getting personal data from perople files to movie files
def enrich_person(person, max_year):
    movies = person["movies"]
    movies = [m for m in movies if m.get("year") < max_year]
    movies = sorted(movies, key=lambda k: k['year'], reverse=True)
    gross = []
    complex_gross = []
    for m in movies:
        gross_usa = m.get("details").get("Gross USA")
        if gross_usa is not None and isinstance(gross_usa, int):
            movie_gross = m.get("details").get("Gross USA")
        else:
            movie_gross = m.get("details").get("Cumulative Worldwide Gross")
        if movie_gross is not None and isinstance(gross_usa, int):
            movie_gross = movie_gross / 1e6
            gross.append(movie_gross)
            complex_gross.append(movie_gross / m.get("place"))

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


def get_genres_binary_array():
    db = get_data(2007, 2017)
    genres = get_list_of_feature(db, "genres")
    usa_gross = get_gross(db)
    genres, usa_gross = remove_nones(genres, usa_gross)
    return set([item for sublist in genres for item in sublist])


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


def parse_dates(orig_dates):
    dates = []
    for i in orig_dates:
        if i is None:
            dates.append(None)
            continue
        parsed = False
        for form in ['%d %B %Y ', '%B %Y ']:
            try:
                dates.append(get_date_ts(time.strptime(i.split("(")[0], form)))
                parsed = True
            except ValueError:
                continue
        if not parsed:
            dates.append(None)
    return dates


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


def get_genre(db, genre_dict):
    genres = {k: np.mean(v) for k, v in genre_dict.items() if len(v) > 0}
    movie_genres = get_list_of_feature(db, "genres")
    avg_movies_gross_by_genre = []
    max_movies_gross_by_genre = []
    for movie in movie_genres:
        genre_gross = [genres[i] for i in movie if i in genres.keys()]
        if len(genre_gross) == 0:
            avg_movies_gross_by_genre.append(0)
            max_movies_gross_by_genre.append(0)
        else:
            avg_movies_gross_by_genre.append(sum(genre_gross) / len(genre_gross))
            max_movies_gross_by_genre.append(max(genre_gross))

    return avg_movies_gross_by_genre, max_movies_gross_by_genre


def append_none(data, func):
    result = []
    for i in data:
        try:
            result.append(func(i))
        except Exception:
            result.append(None)

    return result


def get_person_params(person_list):
    features = ["average_gross", "max_gross", "std_gross", "avg_gradient", "average_complex_gross"]
    params = []
    for feature in features:
        # person_features = []
        # for person in person_list:
        #     if person is None:
        #         person_features.append(None)
        #     else:
        #         person_features.append(person.get(feature))
        person_features = append_none(person_list, lambda person: person.get(feature))
        params.append(person_features)
    return params


def get_all_params(db, genres):
    release_date = get_list_of_details_feature(db, "Release Date")
    dates = parse_dates(release_date)
    cast = get_list_of_feature(db, "cast")
    cast_num = [len(i) for i in cast]
    runtime = get_list_of_details_feature(db, "Runtime")
    usa_gross = get_gross(db)
    avg_movies_gross_by_genre, max_movies_gross_by_genre = get_genre(db, genres)
    avg_cast_age = get_list_of_feature(db, "average_age")
    women_ratio = get_list_of_feature(db, "actress_ratio")
    params = [cast_num, avg_movies_gross_by_genre, max_movies_gross_by_genre, women_ratio, avg_cast_age]
    params += get_person_params(get_list_of_feature(db, "producer_enriched"))
    params += get_person_params(get_list_of_feature(db, "director_enriched"))
    params += get_person_params(get_list_of_feature(db, "writer_enriched"))
    movies_cast = get_list_of_feature(db, "cast_enriched")
    for i in range(15):
        actor_by_index = append_none(movies_cast, lambda actor: actor[i])
        params += get_person_params(actor_by_index)
    return usa_gross, list(zip(*params))


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
def get_linear_fit(db, genres):
    usa_gross, X = get_all_params(db, genres)
    linear = LinearRegression()
    X, usa_gross = remove_nones(X, usa_gross)
    return linear.fit(X=X, y=usa_gross)


def get_linear_predict(db, linear, genres):
    usa_gross, X = get_all_params(db, genres)
    X, usa_gross = remove_nones(X, usa_gross)
    predicts = linear.predict(X)
    return mean_squared_error(predicts, usa_gross)


def create_all_histograms():
    db = get_data(2007, 2017)
    release_date = get_list_of_details_feature(db, "Release Date")
    dates = parse_dates(release_date)
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
    db = get_data(2007, 2015)
    genres = get_genre_dict(db)
    linear = get_linear_fit(db, genres)
    # with open(path_to_data + "db_test.json") as f:
    #     db = json.load(f)
    db = get_data(2015, 2017)
    print(get_linear_predict(db, linear, genres))


get_data(2007,2017)
