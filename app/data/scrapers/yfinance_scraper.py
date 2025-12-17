import time
import json

import requests
from bs4 import BeautifulSoup
import selenium.webdriver as webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

from app.data.scrapers.base import Article, Scraper, datetime

class YFinanceScraper(Scraper):
    """Scraper for the Yahoo Finance website"""

    def __init__(self):
        super().__init__("YFinance_Scraper")

        # browser agent headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.initialize_cookies()

        self.log("initialized")

    def initialize_cookies(self):
        """Accepts cookies and set them for the session"""
        self.log("Initializing cookies...")

        try:
            options = Options()
            options.add_argument("--headless=new")
            driver = webdriver.Chrome(options=options)

            # fetching a particular news site like this will always trigger a consent page (i hope)
            driver.get("https://finance.yahoo.com/news/rocket-lab-q3-earnings-call-053227894.html")

            scroll_down = driver.find_element(By.ID, "scroll-down-btn")
            scroll_down.click()
            accept_cookies = driver.find_element(By.CLASS_NAME, "accept-all")
            accept_cookies.click()

            time.sleep(1)
            cookies = driver.get_cookies()
            self.cookies = {c["name"]: c["value"] for c in cookies}
            self.session.cookies.update(self.cookies)
        except Exception as e:
            self.log(f"Failed to initialize cookies: {e}")

    def scrape(self, keyword: str, from_: datetime) -> list[Article]:
        self.log(f"scraping for keyword: {keyword} from: {from_.isoformat()}")

        # Initial fetch from html page
        response = self.session.get(f"https://uk.finance.yahoo.com/quote/{keyword}/news/")
        if not response.ok:
            self.log(f"Failed fetching: {response.text}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")

        # find the content from the script tags
        scripts = soup.find_all("script", {"type": "application/json"})

        content = None
        for script in scripts:
            if "paginationString" in script.get_text():
                content = script.get_text()
                break

        if not content:
            self.log(f"Failed to retrieve content for {keyword}")
            return []

        data = json.loads(json.loads(content)["body"])
        scraped_articles = []

        finished_scraping = False
        while not finished_scraping:

            # sort articles and fetch until from_
            fetched_articles: list = data["data"]["tickerStream"]["stream"]
            fetched_articles = sorted(fetched_articles, key=lambda x: x["content"]["pubDate"], reverse=True)

            for article in fetched_articles:
                # only get internal articles
                if "yahoo" not in article["content"]["canonicalUrl"]["url"][:30]:
                    continue

                self.log(f"{article["content"]["title"]}: {article["content"]["pubDate"]}")

                if article["content"]["pubDate"] < from_.isoformat():
                    # we scraped enough
                    finished_scraping = True
                    break

                scraped_articles.append(article["content"]["canonicalUrl"]["url"])

            if finished_scraping:
                break

            # get next page using current paginationString
            payload = {
                "payload": {
                    "gqlVariables": {
                        "tickerStream": {
                            "pagination": {
                                "uuids": data["data"]["tickerStream"]["pagination"]["uuids"]
                            }
                        }
                    }
                },
                "serviceConfig": {
                    "count": 250,
                    "snippetCount": 20
                }
            }

            self.log("Fetching next page")
            response = self.session.post(f"https://uk.finance.yahoo.com/xhr/ncp?location=GB&queryRef=newsAll&serviceKey=ncp_fin&listName={keyword}-news&lang=en-GB&region=GB", json=payload)

            if not response.ok:
                self.log(f"Failed fetching next page: {response.text}")
                break

            data = json.loads(response.text)

        self.log(f"Found {len(scraped_articles)} articles")

        # get the individual articles
        articles: list[Article] = []
        for url in scraped_articles:
            response = self.session.get(url)

            if not response.ok:
                self.log(f"Failed fetching: {url}")
                continue

            soup = BeautifulSoup(response.text, "html.parser")

            author = soup.find("div", {"class": "byline-attr-author"}).get_text(strip=True)
            timestamp = soup.find("time", {"class": "byline-attr-meta-time"}).get("datetime")
            timestamp = datetime.fromisoformat(timestamp)
            title = soup.find("title").get_text()

            body_div = soup.find("div", {"class": "bodyItems-wrapper"})
            body = body_div.get_text(separator=" ", strip=True)

            articles.append(Article(
                title=title,
                body=body,
                url=url,
                author=author,
                source="Yahoo Finance",
                timestamp=timestamp
            ))


        return articles

if __name__ == "__main__":
    scraper = YFinanceScraper()
    print(scraper.scrape("RKLB", datetime.fromisoformat("2025-12-10")))
