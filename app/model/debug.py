# Debug test - add this temporarily
from main.py import extract_entities_with_positions, get_ticker_lookup

test_text = """
Autodesk Sees Customers Taking Longer on Software Decisions

Autodesk said enterprise customers are extending evaluation periods for 
large software purchases amid economic uncertainty. Revenue grew modestly 
but new bookings were softer than hoped.

The design software company maintained its annual guidance while 
acknowledging the near-term environment remains challenging. Cloud 
transition continues to progress as expected.

Tech sector analysts noted this is consistent with broader trends of 
cautious IT spending rather than company-specific issues.
"""

# Check ticker lookup
_, ticker_lookup_lower, known_tickers = get_ticker_lookup()
print("Is 'autodesk' in lookup?", 'autodesk' in ticker_lookup_lower)
print("What does it map to?", ticker_lookup_lower.get('autodesk'))
print("Is 'ADSK' in known_tickers?", 'ADSK' in known_tickers)

# Check entity extraction
entities = extract_entities_with_positions(test_text)
print("\nEntities found:")
for e in entities:
    print(f"  {e.text} -> {e.ticker} ({e.entity_type}) relevance={e.relevance}")