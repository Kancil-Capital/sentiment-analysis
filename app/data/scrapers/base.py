from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()  # Also print to console
    ]
)

@dataclass
class Article:
    title: str
    body: str
    url: str
    author: str | None
    source: str
    timestamp: datetime

    def __str__(self) -> str:
        return f"""
        Article(
            title: {self.title},
            body: {self.body[:100]}...,
            url: {self.url},
            author: {self.author},
            source: {self.source},
            timestamp: {self.timestamp.isoformat()}
        )
        """

    def __repr__(self) -> str:
        return self.__str__()

    def to_json(self) -> dict:
        return {
            "title": self.title,
            "body": self.body,
            "url": self.url,
            "author": self.author,
            "source": self.source,
            "timestamp": self.timestamp.isoformat()
        }

class Scraper(ABC):
    """Base class for scraper implementations of various news sources"""

    def __init__(self, name: str):
        self.name = name
        self.keyword = ""  # Current keyword being scraped

    @abstractmethod
    def scrape(self, keyword: str, from_: datetime, **kwargs) -> list[Article]:
        """Scrapes articles for a particular keyword"""
        pass

    def log(self, log_string: str):
        kw_string = " (" + self.keyword + ")" if self.keyword else ""
        logging.info(f"{self.name}{kw_string}: {log_string}")

    def error(self, error_string: str):
        kw_string = " (" + self.keyword + ")" if self.keyword else ""
        logging.error(f"{self.name}{kw_string} ERROR: {error_string}")

    def warn(self, warn_string: str):
        logging.warning(f"{self.name} ({self.keyword}) WARNING: {warn_string}")
