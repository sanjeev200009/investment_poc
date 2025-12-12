import os
import json
import random
from datetime import datetime

# Try to import OpenAI/DeepSeek, but have fallback
try:
    from openai import OpenAI
    HAS_AI_LIB = True
except ImportError:
    HAS_AI_LIB = False

def generate_ai_insight(symbol, stock_data, news_data):
    """
    Generate beginner-friendly investment insight using AI or fallback template
    """
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    if api_key and HAS_AI_LIB:
        try:
            # Initialize client based on available API key
            if os.getenv("DEEPSEEK_API_KEY"):
                client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com"
                )
                model = "deepseek-chat"
            else:
                client = OpenAI(api_key=api_key)
                model = "gpt-3.5-turbo"
            
            # Prepare context for AI
            context = f"""
            Stock Symbol: {symbol}
            Company: {stock_data.get('company_name', 'Unknown')}
            Current Price: LKR {stock_data.get('price', 0):,.2f}
            Daily Change: {stock_data.get('change_percent', 0):+.2f}%
            Trading Volume: {stock_data.get('volume', 0):,}
            
            Recent News:
            """
            
            for i, article in enumerate(news_data.get('articles', [])[:3]):
                context += f"\n{i+1}. {article.get('title', 'No title')}"
                context += f"\n   Summary: {article.get('description', 'No summary')[:100]}..."
            
            # Create prompt for beginner investors
            prompt = f"""
            You are an AI investment assistant helping beginner investors in Sri Lanka understand stock market information.
            
            Context: {context}
            
            Please provide a simple, beginner-friendly explanation that:
            1. Explains what the current stock performance means in simple terms
            2. Mentions if the movement looks generally positive, negative, or neutral
            3. Connects any relevant news to the stock's performance
            4. Provides 2-3 key points a beginner should consider
            5. Uses simple language without financial jargon
            6. Includes a disclaimer that this is not financial advice
            
            Format the response in clear paragraphs. Be educational and helpful.
            """
            
            # Call AI API
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful investment assistant for beginner investors in Sri Lanka. Explain everything in simple, clear terms."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            
            return {
                'symbol': symbol,
                'explanation': ai_response,
                'key_points': [
                    "Always do your own research before investing",
                    "Consider your financial goals and risk tolerance",
                    "Diversify your investments across different sectors"
                ],
                'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model_used': model,
                'source': 'AI API'
            }
            
        except Exception as e:
            print(f"AI API Error: {e}")
            return _get_sample_ai_insight(symbol, stock_data, news_data)
    
    else:
        # No API key or library, use sample response
        print("No AI API key found or library not installed, using sample insight")
        return _get_sample_ai_insight(symbol, stock_data, news_data)

def _get_sample_ai_insight(symbol, stock_data, news_data):
    """
    Generate sample AI insight for demonstration
    """
    price = stock_data.get('price', 0)
    change_percent = stock_data.get('change_percent', 0)
    volume = stock_data.get('volume', 0)
    
    # Determine sentiment based on data
    if change_percent > 1:
        sentiment = "positive"
        sentiment_emoji = "📈"
    elif change_percent < -1:
        sentiment = "negative"
        sentiment_emoji = "📉"
    else:
        sentiment = "neutral"
        sentiment_emoji = "➡️"
    
    # Volume analysis
    if volume > 500000:
        volume_desc = "high trading volume, indicating strong investor interest"
    elif volume > 100000:
        volume_desc = "moderate trading activity"
    else:
        volume_desc = "lower trading volume today"
    
    # Generate sample explanation
    explanations = [
        f"""
        ## Understanding {symbol}'s Performance {sentiment_emoji}
        
        **Current Situation**: {symbol} is trading at LKR {price:,.2f}, which is a {change_percent:+.2f}% change from yesterday. This shows a {sentiment} movement in the stock price.
        
        **What This Means for Beginners**:
        - **Price Movement**: A {change_percent:+.2f}% change is considered {"significant" if abs(change_percent) > 2 else "moderate"} in stock market terms
        - **Trading Activity**: The stock has {volume_desc}
        - **Market Context**: Always compare individual stock performance with the overall market trend
        
        **Key Considerations**:
        1. **Don't Panic Over Daily Changes**: Stock prices naturally fluctuate daily
        2. **Look at Long-Term Trends**: Single-day changes are less important than weekly/monthly trends
        3. **Understand the Company**: Research what {stock_data.get('company_name', 'the company')} actually does
        
        **News Impact**: Recent news about the company or sector can affect stock prices. Consider if any major announcements were made.
        
        ⚠️ **Remember**: This analysis is for educational purposes only. Always consult with a financial advisor before making investment decisions.
        """,
        
        f"""
        ## Beginner's Guide to {symbol}'s Stock Activity
        
        **Quick Summary**: {symbol} is currently {"up" if change_percent > 0 else "down"} by {abs(change_percent):.2f}%, trading at LKR {price:,.2f} with {volume:,} shares traded.
        
        **Simple Explanation**:
        - **Stock Price**: Like any product, stock prices change based on supply and demand
        - **Percentage Change**: Shows how much the price moved compared to yesterday
        - **Trading Volume**: Indicates how many investors are buying/selling
        
        **For Sri Lankan Investors**:
        - Check if this stock is in the ASPI (All Share Price Index)
        - Consider the company's sector ({stock_data.get('sector', 'Unknown')})
        - Look at the company's fundamentals (profits, growth, etc.)
        
        **Beginner Tip**: Start by understanding basic financial terms and track a few stocks for a month before considering investment.
        
        ⚠️ **Disclaimer**: This is an AI-generated educational explanation. Not financial advice.
        """
    ]
    
    return {
        'symbol': symbol,
        'explanation': random.choice(explanations),
        'key_points': [
            f"{symbol} shows {sentiment} movement today",
            f"Trading volume: {volume:,} shares",
            "Consider long-term trends, not just daily changes",
            "Research company fundamentals before investing"
        ],
        'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_used': 'Sample Generator',
        'source': 'Sample Data',
        'note': 'This is a sample AI response for demonstration. With API keys, real AI would be used.'
    }