import requests
import json
import os
from bs4 import BeautifulSoup
from multiprocessing import Process
from path_to_data import get_data_path, get_base_path


class Metadata:
    @staticmethod
    def get_soup(url):
        res = requests.get(url)
        if res.status_code != 200:
            raise RuntimeError(url)
        page = res.text
        return BeautifulSoup(page, "html.parser")


class PersonInMovieMetadata(Metadata):

    def __init__(self, tr):
        a = tr.find("a")
        self.url = "https://www.imdb.com" + a['href']
        # if a.text:
        #     self.name = a.text
        # else:
        #     self.name = tr.find("span", {"class": "itemprop"}).text

class MovieMetadata(Metadata):
    def __init__(self, url, is_full=True):
        # try:
        self.base_url = "/".join(url.split("/")[:-1])
        self.soup = self.get_soup(url)
        title = self.soup.find("div", {"class": "title_wrapper"})
        if title is None:
            return
        self.name = title.find("h1").text.split("\xa0")[0]
        self.year = int(title.find("a").text)
        self.details = {}
        self.get_details()
        if is_full:
            self.cast = []
            self.producers = []
            self.directors = []
            self.writers = []
            self.set_cast_and_crew()

            desc = self.soup.find("div", {"id": "titleStoryLine"})
            try:
                self.story = desc.find("span", {"itemprop": "description"}).text
                self.story_url = desc.find("span", {"class": "see-more inline"}).find("a")["href"]
            except Exception:
                pass
            desc = desc.find_all("div", {"class": "see-more inline canwrap"})
            genres = None
            for i in desc:
                h4 = i.find("h4")
                if h4 is not None and h4.text == "Genres:":
                    genres = i
                    break
            self.genres = [a.text for a in genres.find_all("a")]
            try:
                self.keywords = self.get_keywords(desc)
            except AttributeError:
                self.keywords = []

    def get_keywords(self, desc):
        url = desc.find("div", {"itemprop": "keywords"}).find("nobr").find("a")["href"]
        url = self.base_url + "/" + url.split("/")[-1]
        soup = self.get_soup(url)
        table = soup.find("table")
        return [td["data-item-keyword"] for td in table.find_all("td", {"class": "soda sodavote"})]

    def set_cast_and_crew(self):
        cast = self.soup.find("div", {"id": "titleCast"})
        url = cast.find("div", {"class": "see-more"}).find("a")['href']
        bs = self.get_soup(self.base_url + "/" + url)
        creds = bs.find("div", {"id": "fullcredits_content"})
        tables = creds.find_all("table")
        self.append_tr_to_list(tables[0], self.directors)
        self.append_tr_to_list(tables[1], self.writers)
        self.append_tr_to_list(tables[2], self.cast)
        self.append_tr_to_list(tables[3], self.producers)

    def get_details(self):
        details = self.soup.find("div", {"id": "titleDetails"}).find_all("div")
        for div in details:
            try:
                h4 = div.find("h4")
                key = h4.text.strip(":")
                t = div.find("time")
                if h4.next_sibling and not h4.next_sibling == '\n':
                    self.details[key] = h4.next_sibling.strip()
                elif t:
                    self.details[key] = t.text
                else:
                    a = div.find("a")
                    self.details[key] = a.text
            except AttributeError:
                pass

        for k, v in self.details.items():
            try:
                if v[0] == '$':
                    self.details[k] = int(v[1:].replace(",", ""))
                elif k == "Runtime":
                    self.details[k] = int(v.split(" ")[0])
            except Exception:
                pass

    @staticmethod
    def append_tr_to_list(table, wanted_list):
        for tr in table.find_all("tr"):
            try:
                wanted_list.append(PersonInMovieMetadata(tr))
            except AttributeError:
                pass
            except TypeError:
                pass

    def to_json(self):
        dic = self.__dict__
        dic.pop("soup")
        for k, v in dic.items():
            if k in ["cast", "producers", "directors", "writers"]:
                dic[k] = [i.__dict__ for i in v]
        return dic


class MovieOfPersonMetadata(MovieMetadata):

    def __init__(self, url, person_name, person_type):
        MovieMetadata.__init__(self, url, is_full=False)
        self.place = self.find_person_in_movie(person_name, person_type)

    def find_person_in_movie(self, person_name, person_type):
        cast = self.soup.find("div", {"id": "titleCast"})
        url = cast.find("div", {"class": "see-more"}).find("a")['href']
        bs = self.get_soup(self.base_url + "/" + url)
        creds = bs.find("div", {"id": "fullcredits_content"})
        tables = creds.find_all("table")
        d = {"director": 0, "writer": 1, "actor": 2, "producer": 3}
        return self.find_place_in_table(tables[d[person_type]], person_name)

    @staticmethod
    def find_place_in_table(table, name):
        count = 1
        for tr in table.find_all("tr"):
            try:
                person = PersonInMovieMetadata(tr)
            except Exception:
                continue
            if person.name == name or person.name == " " + name + '\n':
                break
            count += 1

        return count


def scrap_ratings(file_list):
    data_path = get_data_path()[:-1]
    for file in file_list:
        print(file)
        with open(file) as f:
            movie = json.load(f)
        url = movie["base_url"]
        try:
            bs = Metadata.get_soup(url)
        except RuntimeError:
            print("url problem")
            continue
        rating = bs.find("span", {"itemprop": "ratingValue"})
        try:
            movie["rating"] = rating.text
            year = file.split("/")[-2]
            with open(f'{data_path}/{year}/{file.split("/")[-1]}', "w+") as f:
                json.dump(movie, f)
        except AttributeError:
            continue


if __name__ == '__main__':
    data_path = get_base_path()
    file_list = []
    for year in range(2012, 2013):
        base_path = data_path + str(year)
        with_rating_path = f'{get_data_path()}{year}'
        file_list += [f'{base_path}/{i}' for i in os.listdir(base_path) if i not in os.listdir(with_rating_path)]
    jump = 100
    print(len(file_list))
    for i in range(0, len(file_list), jump):
        p = Process(target=scrap_ratings, args=(file_list[i:i + jump],))
        p.start()

