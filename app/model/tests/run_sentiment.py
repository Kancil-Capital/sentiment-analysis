from app.model.main import get_sentiment, get_sentiment_batch, extract_entities_with_positions

if __name__ == "__main__":
    import time
    
    print("=" * 70)
    print("SENTIMENT ANALYSIS")
    print("=" * 70)
    
    test_cases = [
    # ===========================================
    # TEST 1: NVDA - Positive but with caveats
    # Expected: NVDA slightly positive (~0.2 to 0.4)
    # ===========================================
    {
        "title": "NVIDIA Reports Solid Quarter Though Supply Constraints Persist",
        "body": """
        NVIDIA posted quarterly results that met expectations, with data center revenue 
        showing continued strength. However, management noted that supply chain constraints 
        continue to limit shipments.
        
        The company maintained its guidance rather than raising it, citing uncertainty 
        around customer inventory levels. Gross margins were stable but did not expand 
        as some analysts had hoped.
        
        Shares were little changed in after-hours trading as investors weighed the 
        solid execution against the cautious tone on near-term visibility.
        """,
        "source": "Reuters",
        "expected_range": (0.1, 0.4)
    },
    {
        "title": "Fed signals patience as markets watch the US economy",
        "body": """
        The Fed said the US economy is cooling. In the US, investors expect rate cuts later.
        Some analysts argue the United States will avoid recession, but the US labor market is mixed.
        """,
        "source": "Reuters"
    }

]
    
    # Call models
    _ = get_sentiment(test_cases[0]['title'], test_cases[0]['body'], test_cases[0]['source'], None)
    
    # Test individually
    print("\n" + "=" * 70)
    print("INDIVIDUAL PROCESSING")
    print("=" * 70)
    
    start = time.time()
    for i, test in enumerate(test_cases, 1):
        t0 = time.time()
        results = get_sentiment(test['title'], test['body'], test['source'], None)
        elapsed = time.time() - t0
        
        print(f"\nTest {i}: {test['title'][:45]}...")
        print(f"Time: {elapsed:.3f}s")
        print(f"{'Entity':<25} {'Sentiment':>10} {'Confidence':>10} {'Label':<8}")
        print("-" * 60)
        
        for entity, sentiment, confidence in results[:6]:
            if sentiment > 0.15:
                label = "POS"
            elif sentiment < -0.15:
                label = "NEG"
            else:
                label = "NEU"
            print(f"{entity:<25} {sentiment:>+10.4f} {confidence:>10.4f} {label}")
    
    print(f"\nTotal individual time: {time.time() - start:.2f}s")
    
    # Batch processing
    print("\n" + "=" * 70)
    print("BATCH PROCESSING")
    print("=" * 70)
    
    start = time.time()
    batch_results = get_sentiment_batch(test_cases)
    batch_time = time.time() - start
    
    print(f"\nBatch time for {len(test_cases)} articles: {batch_time:.2f}s")
    print(f"Average per article: {batch_time/len(test_cases):.3f}s")
    
    for i, (test, results) in enumerate(zip(test_cases, batch_results), 1):
        print(f"\nArticle {i}: {test['title'][:45]}...")
        print(f"{'Entity':<25} {'Sentiment':>10} {'Confidence':>10} {'Label':<8}")
        print("-" * 60)
        
        for entity, sentiment, confidence in results[:5]:
            if sentiment > 0.15:
                label = "POS"
            elif sentiment < -0.15:
                label = "NEG"
            else:
                label = "NEU"
            print(f"{entity:<25} {sentiment:>+10.4f} {confidence:>10.4f} {label}")
    
    print("\n" + "=" * 70)
    print("DONE")


