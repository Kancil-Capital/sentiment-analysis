import json
import requests
from bs4 import BeautifulSoup

from app.data.scrapers.base import Article, Scraper, datetime

class CNBCScraper(Scraper):
    """Scraper for the CNBC website"""

    def __init__(self):
        super().__init__("CNBC_Scraper")

        # browser agent headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

        # url for fetching news summaries
        self.url = "https://api.queryly.com/cnbc/json.aspx"
        self.params = {
            "queryly_key": "31a35d40a9a64ab3",
            "endindex": "0",
            "batchsize": "20",
            "timezoneoffset": "0",
            "sort": "date"
        }
 
        self.log("initialized")

    def get_body(self, html_body: str, url: str, title: str) -> str | None:
        """Helper to get article body from html"""
        soup = BeautifulSoup(html_body, "html.parser")

        if "/video/" in url:
            # video type, we can only get summary
            ld = soup.find_all("script", {"type": "application/ld+json"})
            if not ld:
                self.log(f"Cannot find ld+json for {title}: {url}")
                return None

            body = json.loads(ld[0].get_text())["description"]
            return body

        else:
            # normal article, get full body
            for i in range(2):
                match i:
                    case 0:
                        # look in script articleBodyText
                        script = soup.find_all("script", {"charset": "UTF-8"})
                        if not script or script[0].get_text().find("articleBodyText") == -1:
                            self.log(f"Cannot find articleBodyText for {title}: {url}")
                            continue

                        script_elem = script[0].get_text()
                        var = script_elem.find("articleBodyText")
                        start = script_elem.find(':', var) + 2
                        end = script_elem.find('",', start + 1)
                        body = script_elem[start:end]
                        return body

                    case 1:
                        # look in xyz-data
                        data = soup.find_all("span", {"class": "xyz-data"})
                        if not data:
                            self.log(f"Cannot find xyz-data for {title}: {url}")
                            continue

                        data = [line.strip() for line in data[0].get_text().splitlines() if line.strip()]
                        body = " ".join(data)
                        return body

            return None



    def scrape(self, keyword: str, from_: datetime) -> list[Article]:
        self.log(f"scraping for keyword: {keyword} from: {from_.isoformat()}")

        scraped_articles = []

        # get all article summaries for the keyword
        self.params["query"] = keyword

        finished_scraping = False
        while not finished_scraping:
            response = requests.get(self.url, params=self.params, headers=self.headers)

            if not response.ok:
                self.log(f"Failed fetching: {response.text}")
                break

            results = response.json()["results"]
            for result in results:
                self.log(f"{result["cn:title"]}: {result["datePublished"]}")
                if result["datePublished"] < from_.isoformat():
                    # we already scraped enough
                    finished_scraping = True
                    break

                scraped_articles.append(result)

            # move on to next page
            self.log("Fetching next page")
            self.params["endindex"] += self.params["batchsize"]

        # fetch the bodies for each article
        self.log(f"Found {len(scraped_articles)} articles")

        articles: list[Article] = []
        for art in scraped_articles:
            title = art["cn:title"]
            url = art["url"]
            author = art.get("author")
            timestamp = datetime.fromisoformat(art["datePublished"])

            html_body = requests.get(url, params=self.params, headers=self.headers)

            if not html_body.ok:
                self.log(f"Failed to fetch body for {title}: {html_body.text}")
                continue

            body = self.get_body(html_body.text, url, title)
            if not body:
                self.log(f"Failed to get body for {title}")
                with open(f"{title}.html", "w") as f:
                    f.write(html_body.text)
                continue

            articles.append(Article(
                title=title,
                body=body,
                url=url,
                author=author,
                source="CNBC",
                timestamp=timestamp
            ))

        return articles


if __name__ == "__main__":
    scraper = CNBCScraper()
    print(scraper.scrape("Pharma", datetime.fromisoformat("2025-12-15")))
