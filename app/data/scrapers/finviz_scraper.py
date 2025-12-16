from dateutil import parser
import json
import requests
from bs4 import BeautifulSoup

from app.data.scrapers.base import Article, Scraper, datetime

class FinvizScraper(Scraper):
    """Scraper for the Finviz website"""

    def __init__(self):
        super().__init__("Finviz_Scraper")

        # browser agent headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

        self.log("initialized")

    def scrape(self, keyword: str, from_: datetime) -> list[Article]:
        self.log(f"scraping for keyword: {keyword} from: {from_.isoformat()}")

        # get list of article links
        response = requests.get(f"https://finviz.com/api/quote-news/{keyword}", headers=self.headers)
        if not response.ok:
            self.log(f"Failed fetching: {response.text}")
            return []

        soup = BeautifulSoup(json.loads(response.text)["html"], "html.parser")
        anchors = soup.find_all("a")

        if not anchors:
            self.log(f"No news found for {keyword}")
            return []

        # only get internal news links i.e. starts with /news/
        links = [link["href"] for link in anchors if link["href"].startswith("/news")]
        self.log(f"Found {len(links)} articles")

        # get individual articles
        articles: list[Article] = []
        for link in links:
            url = f"https://finviz.com{link}"

            try:
                html_body = requests.get(url, headers=self.headers)

                if not html_body.ok:
                    self.log(f"Failed to fetch body for: {link}")
                    continue

                soup = BeautifulSoup(html_body.text, "html.parser")
                title = soup.find("title").get_text()

                publish_info = soup.find_all("div", {"class": "grow self-center"})
                body_div = soup.find_all("div", {"class": "text-justify"})

                # get author and timestamp
                sections = publish_info[0].get_text().split("\r\n")
                sections = [s.strip() for s in sections if s.strip()]
                author, timestamp = sections[1].split("|")

                author = author.strip()
                timestamp = timestamp.strip()
                timestamp = parser.parse(timestamp)

                if timestamp < from_:
                    # we scraped enough
                    break

                # get body
                self.log(f"{title}: {timestamp.isoformat()}")
                body = "".join(b.get_text() for b in body_div)

                articles.append(Article(
                    title=title,
                    body=body,
                    url=f"https://finviz{link}",
                    author=author,
                    source="Finviz",
                    timestamp=timestamp
                ))

            except Exception as e:
                self.log(f"Failed to fetch body for {url}: {e}")
                continue

        return articles

if __name__ == "__main__":
    scraper = FinvizScraper()
    print(scraper.scrape("ADSK", datetime.fromisoformat("2025-12-10")))
