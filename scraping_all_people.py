import json
from multiprocessing import Process
import glob
import logging
import numpy as np
from scraping_person_page import PersonMetadata
from scraping_movie_page import MovieMetadata

logger = logging.getLogger(__name__)
logger.setLevel("INFO")


def create_link_list():
    links = {"actor": [], "producer": [], "director": [], "writer": []}
    years = [str(i) for i in range(2007, 2017)]
    for year in years:
        files = glob.glob("DB/{}/*".format(year))
        for file in files:
            if "movies_list" in file:
                continue
            with open(file) as f:
                data = json.load(f)
            try:
                links["actor"] += [i["url"] for i in data["cast"]]
                links["writer"] += [i["url"] for i in data["writers"]]
                links["director"] += [i["url"] for i in data["directors"]]
                links["producer"] += [i["url"] for i in data["producers"]]
            except Exception as e:
                logger.error("Could not read from: {}, {}".format(file, e))
                movie = MovieMetadata(data["base_url"] + "/?ref_=fn_al_tt_1")
                data = movie.to_json()
                with open(file, "w") as f:
                    json.dump(data, f)


def scrap_some_people(start_idx, links_range, person_type, existing):
    # start_idx, links_range = args
    for link in links_range:
        start_idx += 1
        if start_idx in existing:
            continue
        print("Starting iteration {}".format(start_idx))
        try:
            person = PersonMetadata(link, person_type, 2016, 2000)
        except Exception as e:
            print(e)
            print("Failed to scrap: {}".format(link))
            continue

        with open("../scraping/DB/{}/{}_{}".format(person_type, start_idx, person.name.replace('"', '')), "w+") as f:
            json.dump(person.to_json(), f)


if __name__ == '__main__':
    # with open("DB/peoples_links.json", "r") as f:
    #     links = json.load(f)
    # k = "producer"
    # links = ["_".join(link.split("_")[:-1]) + "_t1" for link in links[k]]
    # set_links = list(set(links))
    # existing = glob.glob("DB/{}/*".format(k))
    # existing = [int(i.split("\\")[-1].split("_")[0]) for i in existing]
    # print(len(set_links))

    with open("../scraping/DB/missing_links") as f:
        all_links = json.load(f)
    k = "actor"
    set_links = all_links[k]
    print(len(set_links))
    existing = glob.glob("../scraping/DB/{}/*".format(k))
    existing = [int(i.split("\\")[-1].split("_")[0]) for i in existing]

    jump = 100
    for i in range(1, len(set_links), jump):
        p = Process(target=scrap_some_people, args=(i + 165229, set_links[i:i+jump], k, existing))
        p.start()





