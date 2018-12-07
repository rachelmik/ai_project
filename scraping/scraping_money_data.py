import json
import glob
from scraping.scraping_movie_page import MovieMetadata


years = ["2012"]

def look_numbers(movie_name):
    if movie_name[:3] == "The":
        movie_name = movie_name[4:] + " The"
    movie_name = movie_name.replace(" ", "-").replace("'","")
    bs = MovieMetadata.get_soup("https://www.the-numbers.com/movie/{}#tab=more".format(movie_name))
    # bs = BeautifulSoup("https://www.the-numbers.com/movie/{}#tab=more".format(movie_name))
    table = bs.find("table", id="movie_finances")
    dic = {}
    for tr in table.find_all("tr"):
        # if hasattr(tr, 'class') and tr['class'] == "heading":
        #     continue
        td = tr.find("td")
        if td.text == "Domestic Box Office":
            next_td = tr.find("td", {"class": "data"})
            dic["Gross USA"] = int(next_td.text[1:].replace(",", ""))
        elif td.text == "Worldwide Box Office":
            next_td = tr.find("td", {"class": "data"})
            dic["Cumulative Worldwide Gross"] = int(next_td.text[1:].replace(",", ""))

    return dic

# print(look_numbers("The Wolf of Wall Street"))

for year in years:
    print(year)
    files = glob.glob("DB/{}/*".format(year))
    for file_name in files:
        if "movies_list" in file_name:
            continue
        with open(file_name) as f:
            data = json.load(f)
        if "Gross USA" not in data["details"].keys() or "Cumulative Worldwide Gross" not in data["details"].keys():
            print(file_name)
            try:
                dic = look_numbers(data["name"])
                data["details"].update(dic)
                with open(file_name, "w") as f:
                    json.dump(data, f)
            except RuntimeError:
                print("Did not succeed: ", file_name)

# with open("DB/2013/00069_A Madea Christmas.json") as f:
#     data = json.load(f)
# print("Gross USA" not in data["details"].keys())