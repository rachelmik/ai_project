from datetime import datetime
from scraping.scraping_movie_page import Metadata, MovieOfPersonMetadata


class PersonMetadata(Metadata):

    def __init__(self, url, person_type, max_year, min_year):
        self.soup = self.get_soup(url)
        self.url = url
        header = self.soup.find("h1", {"class": "header"})
        self.name = header.find("span", {"class": "itemprop"}).text
        self.type = person_type
        if person_type == "actor":
            self.check_gender()
        self.movies = []
        self.get_movies(max_year, min_year)
        try:
            self.age = 0
            self.country = ""
            self.get_birth()
        except Exception:
            pass

    def get_movies(self, max_year, min_year):
        filmography = self.soup.find("div", {"id": "filmography"})
        div = filmography.find("div", {"data-category": self.type})
        try:
            filmography = div.find_next_sibling("div")
        except Exception:
            return
        for div in filmography.find_all("div", {"class": "filmo-row"}):
            if "Series" in div.text or "Video" in div.text:
                continue
            film = div.find("a")["href"]
            year = div.find("span", {"class": "year_column"}).text.strip().split("/")[0]
            if not year.isdigit() or int(year) > max_year or int(year) < min_year:
                continue
            try:
                person_type = "actor" if self.type == "actress" else self.type
                movie = MovieOfPersonMetadata("https://www.imdb.com/" + film, self.name, person_type)
                self.movies.append(movie)
            except Exception:
                continue

    def check_gender(self):
        info = self.soup.find("div", id="name-job-categories")
        for a in info.find_all("span"):
            if a.text.strip() == "Actress":
                self.type = "actress"
                break

    def get_birth(self):
        div = self.soup.find("div", id="name-born-info")
        date = div.find("time")
        self.age = datetime.now().year - datetime.strptime(date["datetime"], "%Y-%m-%d").year
        place = date.next_sibling.next_sibling
        self.country = place.text.split(",")[-1].strip()

    def to_json(self):
        dic = self.__dict__
        dic.pop("soup")
        dic["movies"] = [i.to_json() for i in self.movies]
        return dic


if __name__ == '__main__':
    p = PersonMetadata("https://www.imdb.com/name/nm4232885/?ref_=ttfc_fc_cl_t1", "actor", 2015, 2000)
    # print(p.to_json())
    print(p.movies[0].place)
