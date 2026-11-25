"""Download the hourly weather file from the Building Data Genome 2 (BDG2) open dataset (~19 MB).
We use the real outdoor temperature of site 'Fox' (a hot, dry US site) for summers 2016 and 2017."""
from pathlib import Path
import urllib.request

URL = "https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data/weather/weather.csv"


def main():
    out = Path(__file__).resolve().parent.parent / "data"
    out.mkdir(exist_ok=True)
    dest = out / "weather.csv"
    if not dest.exists():
        print("downloading weather.csv ...")
        urllib.request.urlretrieve(URL, dest)
    print("done ->", dest)


if __name__ == "__main__":
    main()
