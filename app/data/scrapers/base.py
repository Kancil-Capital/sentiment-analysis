from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

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
            souce: {self.source},
            timestamp: {self.timestamp.isoformat()}
        )
        """

    def __repr__(self) -> str:
        return self.__str__()

class Scraper(ABC):
    """Base class for scraper implementations of various news sources"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def scrape(self, keyword: str, from_: datetime) -> list[Article]:
        """Scrapes articles for a particular keyword"""
        pass

    def log(self, log_string: str):
        print(f"{self.name}: {log_string}")
