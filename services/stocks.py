import requests
import json
import os
from datetime import datetime
import random

def get_stock_data(symbol):
    """
    Fetch stock data from Finnhub API or use sample data as fallback
    """
    api_key = os.getenv("FINNHUB_API_KEY")
    
    # Map Sri Lankan stock symbols to Finnhub symbols if needed
    # Note: Finnhub might not have all Sri Lankan stocks
    symbol_mapping = {
        "HNB": "HNB",
        "COMB": "COMB",
        "DIAL": "DIAL",
        "JKH": "JKH",
        "LOFC": "LOFC"
    }
    
    finnhub_symbol = symbol_mapping.get(symbol, symbol)
    
    # Try to fetch from API if key exists
    if api_key:
        try:
            # Finnhub quote endpoint
            quote_url = f"https://finnhub.io/api/v1/quote?symbol={finnhub_symbol}&token={api_key}"
            
            # Finnhub company profile endpoint
            profile_url = f"https://finnhub.io/api/v1/stock/profile2?symbol={finnhub_symbol}&token={api_key}"
            
            response = requests.get(quote_url, timeout=10)
            
            if response.status_code == 200:
                quote_data = response.json()
                
                # Get company profile
                profile_response = requests.get(profile_url, timeout=10)
                profile_data = profile_response.json() if profile_response.status_code == 200 else {}
                
                # Format the data
                stock_data = {
                    'symbol': symbol,
                    'company_name': profile_data.get('name', f'{symbol} Company'),
                    'price': quote_data.get('c', 0),
                    'change': quote_data.get('d', 0),
                    'change_percent': quote_data.get('dp', 0),
                    'high': quote_data.get('h', 0),
                    'low': quote_data.get('l', 0),
                    'open': quote_data.get('o', 0),
                    'previous_close': quote_data.get('pc', 0),
                    'volume': random.randint(10000, 1000000),  # Mock volume since Finnhub doesn't provide
                    'sector': profile_data.get('finnhubIndustry', 'Financial Services'),
                    'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'source': 'Finnhub API',
                    'currency': 'LKR',
                    'country': 'Sri Lanka'
                }
                
                # Check if we got valid data (Finnhub returns 0 for unknown symbols)
                if quote_data.get('c') == 0:
                    return _get_sample_stock_data(symbol)
                
                return stock_data
            else:
                # API returned error, use sample data
                return _get_sample_stock_data(symbol)
                
        except Exception as e:
            print(f"API Error: {e}")
            return _get_sample_stock_data(symbol)
    else:
        # No API key, use sample data
        print("No Finnhub API key found, using sample data")
        return _get_sample_stock_data(symbol)

def _get_sample_stock_data(symbol):
    """
    Get sample stock data from JSON file
    """
    try:
        with open('sample_data/sample_stock.json', 'r') as f:
            sample_data = json.load(f)
        
        # Customize based on symbol
        stock_data = sample_data.get(symbol, sample_data['DEFAULT'])
        
        # Update with current symbol and timestamp
        stock_data['symbol'] = symbol
        stock_data['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stock_data['source'] = 'Sample Data (Fallback)'
        
        # Add some random variation for demonstration
        stock_data['price'] = round(stock_data['price'] * random.uniform(0.98, 1.02), 2)
        stock_data['change'] = round(stock_data['price'] - stock_data['previous_close'], 2)
        stock_data['change_percent'] = round((stock_data['change'] / stock_data['previous_close']) * 100, 2)
        
        return stock_data
        
    except Exception as e:
        print(f"Error loading sample data: {e}")
        return {
            'symbol': symbol,
            'company_name': f'{symbol} Company',
            'price': round(random.uniform(100, 500), 2),
            'change': round(random.uniform(-10, 10), 2),
            'change_percent': round(random.uniform(-2, 2), 2),
            'high': round(random.uniform(450, 550), 2),
            'low': round(random.uniform(90, 150), 2),
            'open': round(random.uniform(100, 120), 2),
            'previous_close': round(random.uniform(95, 115), 2),
            'volume': random.randint(50000, 500000),
            'sector': 'Financial Services',
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'source': 'Generated Sample',
            'currency': 'LKR',
            'country': 'Sri Lanka',
            'error': 'Using generated sample data due to API limitations'
        }