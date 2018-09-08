import json
import os
from datetime import datetime
import time
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def get_data(min_year, max_year):
    db = []
    for year in range(min_year, max_year):
        print(year)
        json_list = os.listdir('./DB/' + str(year))
        for json_file in json_list:
            with open('./DB/' + str(year)+'/'+json_file) as f:
                movie = json.load(f)
                db.append(movie)
    return db


def get_people(person_type):
    people = []
    bad_files = []
    json_list = os.listdir('./DB/{}'.format(person_type))
    count = 0
    for json_file in json_list:
        try:
            if count % 100 == 0:
                print(count)
            count += 1
            with open('./DB/{}/{}'.format(person_type,json_file)) as f:
                person = json.load(f)
            person = enrich_person(person)
            people.append(person)
            with open('./DB/{}/{}'.format(person_type, json_file), "w") as f:
                json.dump(person, f)
        except Exception as e:
            print(e)
            print(json_file)
            bad_files.append(json_file)
            continue
    print(bad_files)
    return people


def check_len(data, func, size):
    if len(data) > size:
        return func(data)
    return 0


def average(data):
    return check_len(data, lambda x: sum(x) / len(x), 0)


def enrich_person(person, max_year):
    movies = person["movies"]
    movies = [m for m in movies if m.get("year") < max_year]
    movies = sorted(movies, key=lambda k: k['year'], reverse=True)
    gross = []
    complex_gross = []
    for m in movies:
        if m.get("details").get("Gross USA") is not None:
            movie_gross = m.get("details").get("Gross USA")
        else:
            movie_gross = m.get("details").get("Cumulative Worldwide Gross")
        if movie_gross is not None:
            movie_gross = movie_gross / 1e6
            gross.append(movie_gross)
            complex_gross.append(movie_gross / m.get("place"))

    gross = gross[:5]
    complex_gross = complex_gross[:5]
    num_of_movies = len(gross)
    person["average_gross"] = sum(gross) / num_of_movies if num_of_movies != 0 else 0
    person["max_gross"] = max(gross) if num_of_movies != 0 else 0
    if num_of_movies > 1:
        gradient = list(np.gradient(gross))
        person["gradient"] = gradient
        person["avg_gradient"] = sum(gradient) / len(gradient)
    person["average_complex_gross"] = sum(complex_gross) / num_of_movies  if num_of_movies != 0 else 0
    return person


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
    return [i / 10**6 for i in gross]


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


def create_hist(values_dic, is_big):
    hist = {k: sum(v) / len(v) for k, v in values_dic.items() if len(v) > 0}
    hist = sorted(hist.items(), key=lambda x: x[0])
    fig, ax = plt.subplots()
    index = np.arange(len(hist))
    rects1 = ax.bar(index, [i[1] for i in hist])
    if is_big:
        ax.set_xticks(np.arange(9)*5 - 0.5)
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
        hist[bins[i]] = [v for c, v in zip(param, usa_gross) if bins[i] <= c < bins[i+1]]
    create_hist(hist, True)


def create_bars(db, param):
    usa_gross = get_gross(db)
    param, usa_gross = remove_nones(param, usa_gross)
    hist = {b: [] for b in set(param)}
    for p, g in zip(param, usa_gross):
        hist[p].append(g)
    create_hist(hist, False)


def get_all_params(db):
    release_date = get_list_of_details_feature(db, "Release Date")
    dates = parse_dates(release_date)
    cast = get_list_of_feature(db, "cast")
    cast = [len(i) for i in cast]
    runtime = get_list_of_details_feature(db, "Runtime")
    usa_gross = get_gross(db)
    genres = get_genre_dict(db)
    genres = {k: sum(v) / len(v) for k, v in genres.items() if len(v) > 0}
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
    return usa_gross, list(zip(cast, avg_movies_gross_by_genre, max_movies_gross_by_genre))


def get_linear_fit(db):
    usa_gross, X = get_all_params(db)
    linear = LinearRegression()
    X, usa_gross = remove_nones(X, usa_gross)
    return linear.fit(X=X, y=usa_gross)


def get_linear_predict(db, linear):
    usa_gross, X = get_all_params(db)
    X, usa_gross = remove_nones(X, usa_gross)
    predicts = linear.predict(X)
    return mean_squared_error(predicts, usa_gross)

# db = get_data(2007, 2017)
# release_date = get_list_of_details_feature(db, "Release Date")
# dates = parse_dates(release_date)
# cast = get_list_of_feature(db, "cast")
# cast = [len(i) for i in cast]
# year = get_list_of_feature(db, "year")
# runtime = get_list_of_details_feature(db, "Runtime")
# usa_gross = get_gross(db)


# create_bars(db, year)
# create_range_hist(db, dates)
# genres = get_genre_dict(db)
# print(genres)
# create_hist(genres, False)

# db = get_data(2007, 2015)
# linear = get_linear_fit(db)
# db = get_data(2015, 2017)
# m = [movie["base_url"] for movie in db if movie.get("genres") is None]
# print(m)
# print(len(m))
# print(get_linear_predict(db, linear))

# get_people("actor2")
# p_list = ['10605_Abhijat Joshi']
# for p in p_list:
#     with open("DB/writer/" + p) as j:
#         person = json.load(j)
#         # try:
#         person = enrich_person(person, 2014)
#         print(person)
#         # except Exception as e:
#         #     print(e)
#         #     continue
#     with open("DB/writer/" + p, "w") as j:
#         json.dump(person, j)

