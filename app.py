import streamlit as st
import json
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from services.stocks import get_stock_data
from services.news import get_news_data
from services.ai_agent import generate_ai_insight
import matplotlib.pyplot as plt
import joblib

# Load environment variables
load_dotenv()

# Load trained model (cached)
@st.cache_resource
def load_trained_model():
    """Load trained model and metadata"""
    try:
        model_data = joblib.load('models/simple_model.pkl')
        with open('models/metadata.json') as f:
            metadata = json.load(f)
        return model_data, metadata
    except FileNotFoundError:
        return None, None

# Load sample data
@st.cache_data
def load_sample_data():
    """Load sample stock data for exploration"""
    try:
        with open('sample_data/sample_stock.json') as f:
            stocks = json.load(f)
        data = []
        for symbol, info in stocks.items():
            data.append({
                'Symbol': symbol,
                'Price (LKR)': info.get('price', 0),
                'Volume': info.get('volume', 0),
                'Change %': info.get('change_percent', 0),
                'Sector': info.get('sector', 'N/A'),
                'Market Cap': info.get('market_cap', 0)
            })
        return pd.DataFrame(data)
    except:
        return None

# Load model on startup
try:
    model_data, model_metadata = load_trained_model()
except Exception as e:
    model_data, model_metadata = None, None

# Page configuration
st.set_page_config(
    page_title="AI Investment Assistant PoC",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #60A5FA;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 0 10px rgba(96, 165, 250, 0.3);
    }
    .section-header {
        font-size: 1.5rem;
        color: #93C5FD;
        margin-top: 2rem;
        border-bottom: 2px solid #3B82F6;
        padding-bottom: 0.5rem;
    }
    .stock-card {
        background-color: #1E293B;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #3B82F6;
        color: #E2E8F0;
    }
    .news-card {
        background-color: #1F2937;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #FBBF24;
        color: #F3F4F6;
    }
    .ai-response {
        background-color: #064E3B;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #10B981;
        color: #D1FAE5;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📈 AI-Powered Investment Assistant PoC</h1>', unsafe_allow_html=True)
st.markdown("### For Beginner Investors in Sri Lanka")
st.markdown("---")

# ============================================================================
# FEATURES TABS (Data Exploration, Visualizations, etc.)
# ============================================================================
feature_tabs = st.tabs(["🏠 Dashboard", "🔍 Data Exploration", "📊 Visualizations", "🔮 Predictions", "📈 Model Performance"])

with feature_tabs[0]:  # Dashboard
    st.markdown("**Welcome to the AI Investment Assistant!**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Stocks Tracked", "5")
    with col2:
        model_status = "✅ Loaded" if model_data else "⚠️ Demo Mode"
        st.metric("🤖 ML Model", model_status)
    with col3:
        st.metric("📈 Data Points", "500+")

with feature_tabs[1]:  # Data Exploration
    st.markdown("**📊 Dataset Overview**")
    df = load_sample_data()
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Records", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Sectors", df['Sector'].nunique())
        with col4:
            st.metric("Avg Price", f"LKR {df['Price (LKR)'].mean():.0f}")
        
        st.markdown("**🔎 Filter Data**")
        col1, col2 = st.columns(2)
        with col1:
            selected_sectors = st.multiselect("Sector:", df['Sector'].unique(), default=df['Sector'].unique())
        with col2:
            price_range = st.slider("Price Range (LKR):", 
                                   float(df['Price (LKR)'].min()), 
                                   float(df['Price (LKR)'].max()),
                                   (float(df['Price (LKR)'].min()), float(df['Price (LKR)'].max())))
        
        filtered_df = df[(df['Sector'].isin(selected_sectors)) & 
                        (df['Price (LKR)'] >= price_range[0]) & 
                        (df['Price (LKR)'] <= price_range[1])]
        
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)
        
        st.markdown("**📋 Summary Stats**")
        st.dataframe(filtered_df[['Price (LKR)', 'Volume', 'Change %']].describe(), use_container_width=True)

with feature_tabs[2]:  # Visualizations
    st.markdown("**📈 Stock Charts**")
    df = load_sample_data()
    if df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Chart 1: Stock Prices**")
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ['#10B981' if x > 0 else '#EF4444' for x in df['Change %']]
            ax.barh(df['Symbol'], df['Price (LKR)'], color=colors, alpha=0.8)
            ax.set_xlabel('Price (LKR)')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("**Chart 2: Price Changes**")
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ['#10B981' if x > 0 else '#EF4444' for x in df['Change %']]
            ax.bar(df['Symbol'], df['Change %'], color=colors, alpha=0.8)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax.set_ylabel('Change (%)')
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("**Chart 3: Trading Volume**")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(df['Symbol'], df['Volume'], color='#3B82F6', alpha=0.8)
        ax.set_ylabel('Volume')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}K'))
        plt.tight_layout()
        st.pyplot(fig)

with feature_tabs[3]:  # Predictions
    st.markdown("**🔮 Stock Price Prediction**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Input Features:**")
        current_price = st.number_input("Current Price (LKR):", 10.0, 1000.0, 100.0, 5.0)
        volume = st.number_input("Volume:", 1000, 10000000, 100000, 10000)
        change = st.slider("Change %:", -10.0, 10.0, 0.0, 0.1)
        predict_btn = st.button("🎯 Predict", use_container_width=True)
    
    with col2:
        st.markdown("**Prediction Results:**")
        if predict_btn:
            if model_data:
                try:
                    features = np.array([[current_price, volume, change, current_price, volume]])
                    means = np.array(model_data['means'])
                    stds = np.array(model_data['stds'])
                    weights = np.array(model_data['weights'])
                    features_norm = (features - means) / (stds + 1e-8)
                    features_with_bias = np.hstack([np.ones((1, 1)), features_norm])
                    predicted_price = (features_with_bias @ weights)[0]
                    change_pct = ((predicted_price - current_price) / current_price * 100) if current_price > 0 else 0
                    st.metric("Predicted Price", f"LKR {predicted_price:.2f}", f"{change_pct:+.2f}%")
                    st.metric("Confidence", f"{model_metadata.get('r2_score', 0):.2%}")
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
            else:
                st.warning("Model not available - using demo prediction")

with feature_tabs[4]:  # Model Performance
    st.markdown("**📈 Model Performance Metrics**")
    if model_data and model_metadata:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² Score", f"{model_metadata.get('r2_score', 0):.4f}")
        with col2:
            st.metric("MAE", f"LKR {model_metadata.get('mae', 0):.2f}")
        with col3:
            st.metric("RMSE", f"LKR {model_metadata.get('rmse', 0):.2f}")
        with col4:
            st.metric("Samples", model_metadata.get('samples', 0))
        
        st.markdown("**Model Info:**")
        st.write(f"Algorithm: Linear Regression | Features: 5 | Status: ✅ Ready")
        
        # Performance chart
        fig, ax = plt.subplots(figsize=(10, 3))
        metrics = ['R² Score', 'MAE/50', 'RMSE/50']
        values = [model_metadata.get('r2_score', 0), 
                 min(1, model_metadata.get('mae', 0)/50),
                 min(1, model_metadata.get('rmse', 0)/50)]
        ax.barh(metrics, values, color=['#10B981', '#3B82F6', '#F59E0B'], alpha=0.8)
        ax.set_xlim(0, 1)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Model not available - using demo metrics")

st.markdown("---")

# Sidebar for information
with st.sidebar:
    st.header("ℹ️ About This PoC")
    st.info("""
    This Proof of Concept demonstrates:
    1. **Data Collection**: Fetching real-time stock data
    2. **News Analysis**: Gathering market news
    3. **AI Processing**: Generating beginner-friendly explanations
    4. **Fallback System**: Using sample data when APIs fail
    """)
    
    st.header("🎯 Sample Stocks")
    st.code("""
    HNB - Hatton National Bank
    COMB - Commercial Bank
    DIAL - Dialog Axiata
    JKH - John Keells Holdings
    LOFC - Lanka IOC
    """)
    
    # API Status
    st.header("🔧 API Status")
    finnhub_key = os.getenv("FINNHUB_API_KEY")
    newsapi_key = os.getenv("NEWSAPI_API_KEY")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Finnhub API", "✅ Ready" if finnhub_key else "❌ Not Set")
    with col2:
        st.metric("NewsAPI", "✅ Ready" if newsapi_key else "❌ Not Set")

# Main input section
col1, col2 = st.columns([3, 1])
with col1:
    stock_symbol = st.text_input(
        "Enter a Sri Lankan Stock Symbol:",
        placeholder="e.g., HNB, COMB, DIAL",
        help="Use CSE stock symbols (3-4 letters)"
    )
with col2:
    st.write("")  # Spacing
    st.write("")  # Spacing
    analyze_button = st.button("🔍 Analyze Stock", type="primary", use_container_width=True)

# Initialize session state for results
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'news_data' not in st.session_state:
    st.session_state.news_data = None
if 'ai_insight' not in st.session_state:
    st.session_state.ai_insight = None

if analyze_button and stock_symbol:
    # Convert to uppercase for consistency
    stock_symbol = stock_symbol.upper().strip()
    
    if not stock_symbol:
        st.warning("Please enter a stock symbol")
        st.stop()
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📊 Stock Data", "📰 Market News", "🤖 AI Explanation"])
    
    with tab1:
        st.markdown('<h3 class="section-header">Stock Information</h3>', unsafe_allow_html=True)
        
        with st.spinner(f"Fetching data for {stock_symbol}..."):
            try:
                stock_data = get_stock_data(stock_symbol)
                st.session_state.stock_data = stock_data
                
                if stock_data.get('error'):
                    st.error(f"Error: {stock_data.get('error')}")
                    st.info("⚠️ Using sample data for demonstration")
                    
                    # Display sample data structure
                    with st.expander("View Sample Data Structure"):
                        st.json(stock_data)
                else:
                    # Display stock information in cards
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Current Price",
                            f"LKR {stock_data.get('price', 'N/A'):,.2f}",
                            f"{stock_data.get('change_percent', 0):+.2f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "Day High",
                            f"LKR {stock_data.get('high', 'N/A'):,.2f}",
                        )
                    
                    with col3:
                        st.metric(
                            "Day Low",
                            f"LKR {stock_data.get('low', 'N/A'):,.2f}",
                        )
                    
                    with col4:
                        st.metric(
                            "Volume",
                            f"{stock_data.get('volume', 0):,}",
                        )
                    
                    # Additional stock info
                    st.markdown('<div class="stock-card">', unsafe_allow_html=True)
                    st.write(f"**Company:** {stock_data.get('company_name', 'N/A')}")
                    st.write(f"**Sector:** {stock_data.get('sector', 'Financial Services')}")
                    st.write(f"**Last Updated:** {stock_data.get('last_updated', 'N/A')}")
                    st.write(f"**Data Source:** {stock_data.get('source', 'Sample Data')}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error fetching stock data: {str(e)}")
                st.info("The system will use fallback sample data for the demonstration")
    
    with tab2:
        st.markdown('<h3 class="section-header">Recent Market News</h3>', unsafe_allow_html=True)
        
        with st.spinner("Gathering relevant news..."):
            try:
                news_data = get_news_data(stock_symbol)
                st.session_state.news_data = news_data
                
                if news_data.get('error'):
                    st.warning(f"News API Error: {news_data.get('error')}")
                    st.info("Displaying sample news articles for demonstration")
                
                articles = news_data.get('articles', [])
                
                if not articles:
                    st.info("No recent news found for this stock.")
                else:
                    for i, article in enumerate(articles[:5]):  # Show max 5 articles
                        st.markdown('<div class="news-card">', unsafe_allow_html=True)
                        st.subheader(f"{i+1}. {article.get('title', 'No Title')}")
                        st.write(f"**Source:** {article.get('source', 'Unknown')}")
                        st.write(f"**Published:** {article.get('published_at', 'N/A')}")
                        st.write(f"**Summary:** {article.get('description', 'No summary available')}")
                        
                        if article.get('url'):
                            st.markdown(f"[Read Full Article]({article['url']})")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"Error fetching news: {str(e)}")
    
    with tab3:
        st.markdown('<h3 class="section-header">AI-Powered Explanation & Predictions</h3>', unsafe_allow_html=True)
        
        # Show model status
        col1, col2 = st.columns([3, 1])
        with col1:
            try:
                if model_data is not None:
                    st.success("✅ ML Model Loaded - Predictions Available")
                else:
                    st.warning("⚠️ ML Model Not Available - Using AI Analysis Only")
            except:
                st.warning("⚠️ ML Model Not Available - Using AI Analysis Only")
        
        with st.spinner("Generating beginner-friendly insights..."):
            try:
                # Get data from session state or fetch again
                stock_data = st.session_state.stock_data or get_stock_data(stock_symbol)
                news_data = st.session_state.news_data or get_news_data(stock_symbol)
                
                # Show ML Prediction if model is available
                try:
                    if model_data is not None:
                        st.markdown('<h4>🤖 Machine Learning Prediction</h4>', unsafe_allow_html=True)
                        try:
                            # Prepare features
                            current_price = stock_data.get('price', 100)
                            volume = stock_data.get('volume', 100000)
                            change = stock_data.get('change_percent', 0)
                            
                            features = np.array([[
                                current_price,
                                volume,
                                change,
                                current_price,  # price MA
                                volume           # volume MA
                            ]])
                            
                            # Normalize features
                            means = np.array(model_data['means'])
                            stds = np.array(model_data['stds'])
                            weights = np.array(model_data['weights'])
                            
                            features_norm = (features - means) / (stds + 1e-8)
                            features_with_bias = np.hstack([np.ones((1, 1)), features_norm])
                            prediction = (features_with_bias @ weights)[0]
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Current Price",
                                    f"LKR {current_price:,.2f}"
                                )
                            with col2:
                                st.metric(
                                    "Predicted Price",
                                    f"LKR {prediction:,.2f}"
                                )
                            with col3:
                                change_pct = ((prediction - current_price) / current_price * 100) if current_price > 0 else 0
                                st.metric(
                                    "Expected Change",
                                    f"{change_pct:+.2f}%"
                                )
                            
                            st.info(f"📊 Model Confidence: {model_metadata.get('r2_score', 0):.2%}")
                            
                        except Exception as e:
                            st.error(f"Prediction error: {str(e)}")
                except Exception as e:
                    st.warning(f"Model prediction unavailable: {str(e)}")
                
                # AI Analysis
                st.markdown('<h4>💡 AI Analysis & Recommendation</h4>', unsafe_allow_html=True)
                ai_insight = generate_ai_insight(stock_symbol, stock_data, news_data)
                st.session_state.ai_insight = ai_insight
                
                if ai_insight.get('error'):
                    st.error(f"AI Service Error: {ai_insight.get('error')}")
                    
                    # Provide fallback explanation
                    st.markdown('<div class="ai-response">', unsafe_allow_html=True)
                    st.write("### 📝 Fallback Explanation (Sample)")
                    st.write("""
                    **What this stock performance means:**
                    The stock is currently trading with moderate activity. For beginner investors, 
                    it's important to look at both the price movement and trading volume.
                    
                    **Key things to consider:**
                    1. **Price Stability**: Check if the price has been consistent
                    2. **Trading Volume**: Higher volume means more investor interest
                    3. **Market Context**: Consider overall market conditions
                    
                    **Remember**: Always do your own research and consider consulting with a financial advisor before making investment decisions.
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="ai-response">', unsafe_allow_html=True)
                    st.write(f"### 🤖 AI Analysis for {stock_symbol}")
                    st.write(ai_insight.get('explanation', 'No explanation generated'))
                    
                    # Display key points if available
                    if 'key_points' in ai_insight:
                        st.write("**📋 Key Points:**")
                        for point in ai_insight['key_points']:
                            st.write(f"• {point}")
                    
                    # Display disclaimer
                    st.markdown("---")
                    st.caption("⚠️ **Disclaimer**: This is an AI-generated explanation for educational purposes only. Not financial advice.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show raw data for transparency
                    with st.expander("View Raw AI Response"):
                        st.json(ai_insight)
                        
            except Exception as e:
                st.error(f"Error generating AI insight: {str(e)}")

elif analyze_button and not stock_symbol:
    st.warning("Please enter a stock symbol to analyze")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
    <p>Final Year Project PoC | BIT (Hons) Network & Mobile Computing | Horizon Campus</p>
    <p>This Proof of Concept demonstrates technical feasibility only. Not for actual investment decisions.</p>
</div>
""", unsafe_allow_html=True)