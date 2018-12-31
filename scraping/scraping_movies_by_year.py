import json
from scraping.scraping_movie_page import MovieMetadata
from path_to_data import get_data_path
import logging


def make_year_pages_list(year=None):
    file_name = get_data_path() + "{}/movies_list.json".format(year)
    try:
        with open(file_name, "r") as f:
            return json.load(f)
    except IOError:
        pass
    pages_links = []
    # url_preffix = "https://www.imdb.com/search/title?year={}&title_type=feature&sort=boxoffice_gross_us,desc&page="\
    #     .format(year)
    # url_suffix = "&ref_=adv_nxt"
    url_preffix = "https://www.imdb.com/search/title?title_type=feature&year={}&sort=boxoffice_gross_us,desc&start={}&ref_=adv_nxt"
    for page in range(51, 1001, 50):
        if page % 10 == 0:
            print(page)
        url = url_preffix.format(year, page)
        soup = MovieMetadata.get_soup(url)
        page_links = soup.find_all("h3")
        for links in page_links:
            try:
                link = links.find("a")['href']
            except TypeError:
                continue
            if link is not None:
                pages_links.append(link)

    with open(file_name, 'w+') as f:
        json.dump(pages_links, f)

    return pages_links


def build_movies_db():
    log = logging.getLogger(__name__)
    db = []
    base_url = "https://www.imdb.com"
    for year in range(2017, 2018):
        count = 1
        movies_of_2015 = make_year_pages_list(year)
        for link in movies_of_2015:
            print("started iteration number : " + str(count))
            count = count + 1
            url = base_url + link
            try:
                movie = MovieMetadata(url)
            except Exception as e:
                log.error("Error {} in {}".format(e, url))
                continue
            try:
                if movie.details.get("Country") not in ["USA", "UK"] or movie.details.get("Language") != "English":
                    log.error('The follow movie wasnt include baacuse it is out side of US, link: %s', url)
                    continue
                # if movie.details.get("Gross USA") is None and movie.details.get("Cumulative Worldwide Gross") is None:
                #     log.error('The follow movie wasnt include because it does not have gross info, link: %s', url)
                #     continue
            except Exception as e:
                log.error("Error {} in {}".format(e, url))
                continue
            # db.append(movie.to_json())
            file_name = get_data_path() + "{}/{:0>5d}_{}.json".format(year, count, movie.name).replace(":", "").replace("?", "")
            try:
                with open(file_name, 'w+') as file:
                    json.dump(movie.to_json(), file)
            except Exception:
                log.error("Could not write file {}".format(file_name))

    return db


build_movies_db()
