from app.model.main import get_sentiment, get_sentiment_batch, extract_entities_with_positions

if __name__ == "__main__":
    import time
    
    print("=" * 70)
    print("SENTIMENT ANALYSIS")
    print("=" * 70)
    
    test_cases = test_cases = [
    # ===========================================
    # TEST 1: NVDA - Positive but with caveats
    # Expected: NVDA slightly positive (~0.1 to 0.4)
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
        "expected_range": (0.1, 0.4),
    },

    # ===========================================
    # TEST 2: Multiple US mentions (region multi-find)
    # Expected: United States detected multiple times; sentiment mildly negative/neutral
    # ===========================================
    {
        "title": "Fed signals patience as markets watch the US economy",
        "body": """
        The Fed said the US economy is cooling. In the US, investors expect rate cuts later.
        Some analysts argue the United States will avoid recession, but the US labor market is mixed.
        """,
        "source": "Reuters",
    },

    # ===========================================
    # TEST 3: Cashtag + company name (AAPL)
    # Expected: AAPL positive
    # ===========================================
    {
        "title": "Big tech rally as $AAPL leads gains",
        "body": """
        Apple shares jumped after the company beat earnings expectations.
        In the US, investors rotated back into technology stocks following the Fed meeting.
        """,
        "source": "Reuters",
        "expected_range": (0.2, 0.6),
    },

    # ===========================================
    # TEST 4: Direct ticker repeated (TSLA)
    # Expected: TSLA neutral to slightly negative
    # ===========================================
    {
        "title": "TSLA dips early before TSLA rebounds",
        "body": """
        TSLA shares fell after opening as deliveries disappointed.
        Later in the session, TSLA recovered after analysts said the miss was temporary.
        """,
        "source": "Reuters",
        "expected_range": (-0.2, 0.2),
    },

    # ===========================================
    # TEST 5: PhraseMatcher company names (AMZN)
    # Expected: AMZN slightly positive
    # ===========================================
    {
        "title": "Amazon expands same-day delivery network",
        "body": """
        Amazon said it will expand logistics capacity to support faster shipping.
        The e-commerce giant expects efficiency gains to improve margins over time.
        """,
        "source": "Reuters",
        "expected_range": (0.1, 0.4),
    },

    # ===========================================
    # TEST 6: Multiple companies + masking (AAPL, MSFT)
    # Expected: Both detected separately, no cross-contamination
    # ===========================================
    {
        "title": "Apple and Microsoft face mixed enterprise demand",
        "body": """
        Apple said iPhone demand remains stable, while Microsoft noted softness
        in enterprise cloud spending. Investors compared Apple margins with
        Microsoft’s cloud growth outlook.
        """,
        "source": "Reuters",
    },

    # ===========================================
    # TEST 7: Sector-only article (Energy)
    # Expected: Energy sector detected, neutral/positive
    # ===========================================
    {
        "title": "Oil prices rise as OPEC signals supply discipline",
        "body": """
        Oil prices climbed after OPEC comments on production targets.
        Energy stocks gained while renewable energy names lagged.
        """,
        "source": "Reuters",
    },

    # ===========================================
    # TEST 8: Europe + UK + ECB mapping
    # Expected: EU and UK regions detected
    # ===========================================
    {
        "title": "ECB holds rates steady as European markets rise",
        "body": """
        The European Central Bank kept policy unchanged.
        European stocks advanced, while UK inflation surprised to the upside.
        """,
        "source": "Reuters",
    },

    # ===========================================
    # TEST 9: China / Chinese mapping
    # Expected: China region detected
    # ===========================================
    {
        "title": "Chinese tech stocks rally on policy support",
        "body": """
        Chinese regulators signaled support for capital markets.
        China-focused funds saw renewed investor inflows.
        """,
        "source": "Reuters",
    },

    # ===========================================
    # TEST 10: ASEAN / SEA mapping
    # Expected: SEA region detected
    # ===========================================
    {
        "title": "ASEAN exports slow as Southeast Asia demand softens",
        "body": """
        Southeast Asia exporters warned of slowing global demand.
        ASEAN trade data showed weaker momentum in recent months.
        """,
        "source": "Reuters",
    },
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


