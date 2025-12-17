import os
from dotenv import load_dotenv

load_dotenv()

from supabase import create_client, Client
import yfinance as yf
from datetime import datetime
from app.data.scrapers import CNBCScraper, FinvizScraper, YFinanceScraper

from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import datetime
import yfinance as yf
from supabase import Client

def insert_tickers(tickers: list[str], sb_client: Client) -> list[dict]:
    db_lock = threading.Lock()
    
    # 1. Sequential: Fetch and insert ticker metadata
    print("Fetching ticker metadata...")
    keywords = set()
    for ticker in tickers:
        ticker_metadata = yf.Ticker(ticker).info
        if ticker_metadata == {"trailingPegRatio": None}:
            raise ValueError(f"Invalid ticker: {ticker}")

        keywords.add(ticker_metadata["region"])
        keywords.add(ticker_metadata["sector"])
        keywords.add(ticker_metadata["country"])

        # insert metadata into database
        with db_lock:
            sb_client.table("tickers").insert({
                "symbol": ticker,
                "region": ticker_metadata["region"],
                "sector": ticker_metadata["sector"],
                "country": ticker_metadata["country"]
            }).execute()
    
    print(f"Found keywords: {keywords}")
    
    # 2. Prepare all parallel tasks
    tasks = []
    
    # Keyword news tasks
    for kw in keywords:
        tasks.append(('keyword', kw))
    
    # Ticker news tasks
    scraper_classes = [CNBCScraper, FinvizScraper, YFinanceScraper]
    for ticker in tickers:
        for scraper_class in scraper_classes:
            tasks.append(('ticker', ticker, scraper_class))
    
    print(f"Starting {len(tasks)} parallel scraping tasks...")
    
    # 3. Execute all tasks in parallel
    def process_task(task):
        try:
            if task[0] == 'keyword':
                _, keyword = task
                print(f"Fetching keyword news for: {keyword}")
                
                scraper = CNBCScraper()
                articles = scraper.scrape(keyword, datetime.fromisoformat("2025-01-01"), is_ticker=False)
                
                if articles:
                    with db_lock:
                        sb_client.table("articles").upsert(
                            [a.to_json() for a in articles],
                            on_conflict="url",
                            ignore_duplicates=True
                        ).execute()
                
                with open("setup.log", "a") as f:
                    f.write(f"✓ Completed keyword: {keyword} ({len(articles)} articles)\n")
                
            elif task[0] == 'ticker':
                _, ticker, scraper_class = task
                scraper_name = scraper_class.__name__
                print(f"Fetching ticker news for: {ticker} using {scraper_name}")
                
                scraper = scraper_class()
                articles = scraper.scrape(ticker, datetime.fromisoformat("2025-01-01"), is_ticker=True)
                
                if articles:
                    with db_lock:
                        sb_client.table("articles").upsert(
                            [a.to_json() for a in articles],
                            on_conflict="url",
                            ignore_duplicates=True
                        ).execute()
                
                with open("setup.log", "a") as f:
                    f.write(f"✓ Completed {ticker} via {scraper_name} ({len(articles)} articles)\n")
                
        except Exception as e:
            with open("setup.log", "a") as f:
                f.write(f"Error processing {task}: {e}\n")
    
    # Run with thread pool
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(process_task, tasks)
    
    print("All scraping tasks completed!")

sb = create_client(
    supabase_url=os.environ["SUPABASE_URL"],
    supabase_key=os.environ["SUPABASE_SECRET_KEY"]
)

portfolio = [
    "ADSK", "GE", "GRAB", "CBRE",
    "C", "NVDA", "BABA", "META"
]

try:
    insert_tickers(portfolio, sb)
except Exception as e:
    print(f"Pipeline failed: {e}")
    raise e
