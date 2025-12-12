import requests
import json
import os
from datetime import datetime, timedelta
import random

def get_news_data(symbol):
    """
    Fetch news data from NewsAPI or use sample data as fallback
    """
    api_key = os.getenv("NEWSAPI_API_KEY")
    
    if api_key:
        try:
            # Calculate date for last 7 days
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            # Search for news about the company/stock
            query = f"{symbol} stock OR {symbol} Sri Lanka OR Colombo Stock Exchange"
            
            news_url = f"https://newsapi.org/v2/everything?q={query}&from={from_date}&sortBy=publishedAt&apiKey={api_key}"
            
            response = requests.get(news_url, timeout=10)
            
            if response.status_code == 200:
                news_data = response.json()
                
                if news_data.get('totalResults', 0) > 0:
                    # Format articles
                    articles = []
                    for article in news_data['articles'][:5]:  # Limit to 5 articles
                        articles.append({
                            'title': article.get('title', 'No Title'),
                            'description': article.get('description', 'No description available'),
                            'url': article.get('url', '#'),
                            'source': article.get('source', {}).get('name', 'Unknown'),
                            'published_at': article.get('publishedAt', 'N/A'),
                            'content': article.get('content', '')[:200] + '...' if article.get('content') else 'No content'
                        })
                    
                    return {
                        'symbol': symbol,
                        'total_articles': len(articles),
                        'articles': articles,
                        'source': 'NewsAPI',
                        'last_fetched': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                else:
                    # No news found, use sample data
                    return _get_sample_news_data(symbol)
            else:
                # API error, use sample data
                return _get_sample_news_data(symbol)
                
        except Exception as e:
            print(f"News API Error: {e}")
            return _get_sample_news_data(symbol)
    else:
        # No API key, use sample data
        print("No NewsAPI key found, using sample data")
        return _get_sample_news_data(symbol)

def _get_sample_news_data(symbol):
    """
    Get sample news data from JSON file
    """
    try:
        with open('sample_data/sample_news.json', 'r') as f:
            sample_data = json.load(f)
        
        # Get news for the specific symbol or default
        news_items = sample_data.get(symbol, sample_data['GENERAL'])
        
        # Customize based on symbol
        articles = []
        for i, item in enumerate(news_items[:3]):  # Limit to 3 articles
            # Make it seem more recent
            days_ago = random.randint(1, 5)
            publish_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")
            
            articles.append({
                'title': item['title'].replace('[STOCK]', symbol),
                'description': item['description'].replace('[STOCK]', symbol),
                'url': f"https://example.com/news/{symbol.lower()}-{i+1}",
                'source': random.choice(['Colombo Gazette', 'Daily FT', 'EconomyNext', 'Ada Derana']),
                'published_at': publish_date,
                'content': item['content'].replace('[STOCK]', symbol)
            })
        
        return {
            'symbol': symbol,
            'total_articles': len(articles),
            'articles': articles,
            'source': 'Sample Data (Fallback)',
            'last_fetched': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'note': 'Using sample news data for demonstration'
        }
        
    except Exception as e:
        print(f"Error loading sample news: {e}")
        # Generate minimal fallback
        return {
            'symbol': symbol,
            'total_articles': 1,
            'articles': [{
                'title': f'Market Update for {symbol}',
                'description': f'Recent trading activity for {symbol} shows normal market conditions.',
                'url': '#',
                'source': 'Sample Source',
                'published_at': datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                'content': 'This is sample content for demonstration purposes.'
            }],
            'source': 'Generated Sample',
            'last_fetched': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'error': 'Using generated sample news data'
        }