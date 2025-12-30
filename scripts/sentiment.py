import os
from dotenv import load_dotenv
import logging

load_dotenv()

from supabase import create_client
from app.model.main import get_sentiment

sb = create_client(
    supabase_url=os.getenv("SUPABASE_URL"),
    supabase_key=os.getenv("SUPABASE_SECRET_KEY")
)

log_filename = f"logs/sentiment_setup.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # Also print to console
    ]
)

to_delete = set()
offset = 0
sentiments = []

while True:
    # get articles
    articles: list[dict] = sb.table("articles")\
        .select("*")\
        .limit(1000)\
        .offset(offset)\
        .order("id")\
        .execute().data

    if not articles:
        break

    logging.info(f"Processing articles {articles[0]['id']} to {articles[-1]['id']}")
    offset += 1000
 
    # get sentiments
    for a in articles:
        results = get_sentiment(
            a["title"],
            a["body"],
            a["source"],
            a["author"],
        )

        logging.info(f"Article {a['id']}: {len(results)} sentiments")
 
        if not results:
            to_delete.add(a["id"])
            continue


        for affected_party, score, confidence in results:
            sentiments.append({
                "article_id": a["id"],
                "affected": affected_party,
                "score": score,
                "confidence": confidence
            })

    # add sentiments to database
    if sentiments:
        logging.info(f"Inserting {len(sentiments)} sentiments into database")
        sb.table("sentiments")\
            .insert(sentiments)\
            .execute()

    sentiments.clear()

# delete articles without sentiments
if to_delete:
    logging.info(f"Deleting {len(to_delete)} articles")
    sb.table("articles")\
        .delete()\
        .in_("id", to_delete)\
        .execute()

logging.info("Sentiment setup finished")
