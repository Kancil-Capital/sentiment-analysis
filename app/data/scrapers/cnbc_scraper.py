import json
import time
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

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
            "endindex": 0,
            "batchsize": 100,
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
                self.error(f"Cannot find ld+json for {title}: {url}")
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
                            self.error(f"Cannot find articleBodyText for {title}: {url}")
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
                            self.error(f"Cannot find xyz-data for {title}: {url}")
                            continue

                        data = [line.strip() for line in data[0].get_text().splitlines() if line.strip()]
                        body = " ".join(data)
                        return body

            return None

    def get_ticker_articles(self, ticker: str, from_: datetime) -> list[tuple]:
        """Gets ticker specific news from a dedicated endpoint"""
        self.log("Fetching ticker specific news")
        time_period = (datetime.now() - from_).days
        graphql_url = "https://webql-redesign.cnbcfm.com/graphql"
        graphql_vars = {
            "pageSize": 30,
            "contentType": ["blogpost", "cnbcvideo", "cnbcnewsstory", "livestory"],
            "symbol": ticker,
            "page": 1,
            "timePeriod": str(time_period),
            "sortBy": "dateline",
            "partner": "cnbc01"
        }
        graphql_params = {
            "operationName": "SymbolSearch",
            "variables": json.dumps(graphql_vars),
            "extensions": json.dumps({
                "persistedQuery": {
                    "version": 1,
                    "sha256Hash": "45e411d8c5c566db6396ca67781b4c0f42a3c594e0e6e87c091333db6a9f5a55"
                }
            })
        }

        scraped_articles = []
        finished_scraping = False

        while not finished_scraping:
            response = requests.get(graphql_url, params=graphql_params, headers=self.headers, timeout=(10, 30))

            if not response.ok:
                self.error(f"Failed fetching {graphql_url} page {graphql_vars["page"]}: {response.status_code}")
                break
 
            results = response.json()["data"]["search"]["results"]
            if not results:
                break

            for res in results:
                self.log(f"{res["title"]}: {res["datePublished"]}")

                if res["datePublished"] < from_.isoformat():
                    finished_scraping = True
                    break

                scraped_articles.append((
                    res["title"],
                    res["url"],
                    res["author"][0]["name"] if res["author"] else "",
                    datetime.fromisoformat(res["datePublished"])
                ))

            if finished_scraping:
                break

            # move to next page
            graphql_vars["page"] += 1
            graphql_params["variables"] = json.dumps(graphql_vars)
            self.log("Fetching next page")

        return scraped_articles

    def get_keyword_articles(self, keyword: str, from_: datetime) -> list[tuple]:
        """Gets general keyword news"""
        self.log("Fetching general keyword news")
        scraped_articles = []
        scraped_ids = set()

        # get all article summaries for the keyword
        self.params["query"] = keyword

        finished_scraping = False
        while not finished_scraping:
            previous_id_size = len(scraped_ids)
            time.sleep(2)
            response = requests.get(self.url, params=self.params, headers=self.headers, timeout=(10, 30))

            if not response.ok:
                self.error(f"Failed fetching {self.url} at index {self.params["endindex"]}: {response.status_code}")
                break

            results = response.json()["results"]
            for res in results:
                self.log(f"{res["cn:title"]}: {res["datePublished"]}")
                if res["datePublished"] < from_.isoformat():
                    # we already scraped enough
                    finished_scraping = True
                    break

                scraped_articles.append((
                    res["cn:title"],
                    res["url"],
                    res.get("author", ""),
                    datetime.fromisoformat(res["datePublished"])
                ))
                scraped_ids.add(res["_id"])

            if len(scraped_ids) == previous_id_size:
                # repeat page, don't scrape anymore
                break

            # move on to next page
            self.params["endindex"] += self.params["batchsize"]
            self.log("Fetching next page")

        return scraped_articles

    def fetch_article_body(self, article_data: tuple) -> Article | None:
        """Fetch single article body - to be called in parallel"""
        title, url, author, timestamp = article_data

        try:
            html_body = requests.get(url, headers=self.headers, timeout=(10, 30))

            if not html_body.ok:
                self.error(f"Failed to fetch body for {url}: {html_body.status_code}")
                return None

            body = self.get_body(html_body.text, url, title)
            if not body:
                self.error(f"Failed to extract body for {title}")
                return None

            return Article(
                title=title,
                body=body,
                url=url,
                author=author,
                source="CNBC",
                timestamp=timestamp
            )
        except Exception as e:
            self.error(f"Failed to retrieve body for {url}: {e}")
            return None

    def scrape(self, keyword: str, from_: datetime, **kwargs) -> list[Article]:
        self.log(f"scraping for keyword: {keyword} from: {from_.isoformat()}")
        self.keyword = keyword

        is_ticker = kwargs.get("is_ticker", False)
        scraped_articles = self.get_ticker_articles(keyword, from_) if is_ticker else self.get_keyword_articles(keyword, from_)

        # fetch the bodies for each article in parallel
        self.log(f"Found {len(scraped_articles)} articles")

        # Split into groups
        groups = [scraped_articles[i:i+50] for i in range(0, len(scraped_articles), 50)]
        total_groups = len(groups)

        def process_group(group_data):
            """Sequentially fetch all articles in this group"""
            group, group_idx = group_data
            results = []
            group_size = len(group)

            for idx, article_data in enumerate(group, 1):
                self.log(f"Group {group_idx}/{total_groups} - Article {idx}/{group_size}: {article_data[0]}")
                article = self.fetch_article_body(article_data)
                if article:
                    results.append(article)
            return results

        # Process all groups in parallel
        all_articles = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            group_results = executor.map(process_group, [(g, i+1) for i, g in enumerate(groups)])

        for results in group_results:
            all_articles.extend(results)

        return all_articles

if __name__ == "__main__":
    scraper = CNBCScraper()
    print(scraper.scrape("Pharma", datetime.fromisoformat("2025-12-15")))
