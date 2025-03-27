import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from pypfopt import EfficientFrontier, risk_models, expected_returns

# Custom CSS for styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #F5F6FA;
    }
    
    .header-text {
        font-size: 2.5rem !important;
        color: #1F4173 !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
    }
    
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    .stButton>button {
        background: #1F4173 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(31, 64, 115, 0.2);
    }
    
    .portfolio-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    .positive {
        color: #00B050 !important;
        font-weight: 600;
    }
    
    .negative {
        color: #FF0000 !important;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []

def get_historical_price(ticker, date):
    """Get historical closing price for a given ticker and date"""
    try:
        data = yf.Ticker(ticker).history(
            start=date - pd.Timedelta(days=7),
            end=date + pd.Timedelta(days=1)
        if not data.empty:
            mask = data.index <= pd.to_datetime(date)
            if mask.any():
                return data[mask].iloc[-1]['Close']
        return None
    except Exception as e:
        st.error(f"Error fetching price for {ticker}: {str(e)}")
        return None

def get_live_market_data():
    """Fetch real-time market data for key indicators"""
    market_data = {'nifty': np.nan, 'gold': np.nan, 'bitcoin': np.nan}
    try:
        usdinr = yf.Ticker("USDINR=X").history(period='1d')['Close'].iloc[-1]
        
        # NIFTY 50
        nifty = yf.Ticker("^NSEI").history(period='1d')
        market_data['nifty'] = nifty['Close'].iloc[-1] if not nifty.empty else np.nan
        
        # Gold in INR
        gold = yf.Ticker("GC=F").history(period='1d')
        if not gold.empty:
            market_data['gold'] = gold['Close'].iloc[-1] * usdinr
            
        # Bitcoin in INR
        btc = yf.Ticker("BTC-USD").history(period='1d')
        if not btc.empty:
            market_data['bitcoin'] = btc['Close'].iloc[-1] * usdinr
            
    except Exception as e:
        st.error(f"Market data error: {str(e)}")
    return market_data

def format_currency(value):
    """Format numeric value as INR currency string"""
    if pd.isna(value):
        return "N/A"
    return f"₹{value:,.2f}"

def get_historical_data(tickers):
    """Get historical prices with robust error handling"""
    try:
        data = yf.download(tickers, period='3y', group_by='ticker')
        clean_data = pd.DataFrame()
        
        for ticker in tickers:
            try:
                # Handle different security types
                if isinstance(data.columns, pd.MultiIndex):
                    if ('Adj Close', ticker) in data.columns:
                        clean_data[ticker] = data[('Adj Close', ticker)]
                    else:
                        clean_data[ticker] = data[('Close', ticker)]
                else:
                    clean_data[ticker] = data['Close']
            except Exception as e:
                st.error(f"Could not process {ticker}: {str(e)}")
                continue
                
        return clean_data.dropna(how='all', axis=1)
    except Exception as e:
        st.error(f"Data download failed: {str(e)}")
        return pd.DataFrame()

def main():
    st.set_page_config(page_title="AI Portfolio Optimizer", layout="wide", page_icon="📈")
    st.markdown('<p class="header-text">AI-Powered Portfolio Optimizer</p>', unsafe_allow_html=True)

    # Sidebar inputs
    with st.sidebar:
        st.header("⚙️ Investor Profile")
        risk_level = st.slider("Risk Tolerance (1-10)", 1, 10, 5)
        st.session_state.risk_level = risk_level
        
        st.markdown("---")
        st.header("📥 Add Investment")
        with st.form(key='portfolio_form'):
            ticker_input = st.text_input("Stock Tickers (comma separated)", "TCS.NS,INFY.NS")
            date = st.date_input("Purchase Date", datetime(2020, 1, 1))
            amount = st.number_input("Total Amount (₹)", 1000, 10000000, 100000)
            if st.form_submit_button("Add to Portfolio"):
                raw_tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
                if not raw_tickers:
                    st.error("Please enter at least one ticker")
                else:
                    amount_per = amount / len(raw_tickers)
                    for t in raw_tickers:
                        ticker = t if '.' in t else f"{t}.NS"
                        price = get_historical_price(ticker, date)
                        if price and not np.isnan(price):
                            st.session_state.portfolio.append({
                                'ticker': ticker,
                                'date': date,
                                'amount': amount_per,
                                'purchase_price': price
                            })
                            st.success(f"Added {ticker} with ₹{amount_per:,.2f}")
                        else:
                            st.error(f"Invalid price for {ticker}")

    # Main content
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("📈 Live Market Snapshot")
        market_data = get_live_market_data()
        cols = st.columns(3)
        metrics = [
            ("NIFTY 50", market_data['nifty'], "+1.2%"),
            ("Gold Price", market_data['gold'], "-0.5%"),
            ("Bitcoin", market_data['bitcoin'], "+3.8%")
        ]
        for col, (label, value, delta) in zip(cols, metrics):
            with col:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(label, 
                        f"{value:.2f}" if not pd.isna(value) else "N/A", 
                        delta=delta)
                st.markdown('</div>', unsafe_allow_html=True)

    # Portfolio analysis
    st.header("📊 Portfolio Analysis")
    if not st.session_state.portfolio:
        st.info("💡 Add investments using the sidebar form")
    else:
        holdings_df = pd.DataFrame(st.session_state.portfolio)
        
        # Get current prices
        current_prices = {}
        for ticker in holdings_df['ticker'].unique():
            try:
                current_prices[ticker] = yf.Ticker(ticker).history(period='1d')['Close'].iloc[-1]
            except:
                current_prices[ticker] = np.nan
        
        # Calculate holdings values
        holdings_df['current_price'] = holdings_df['ticker'].map(current_prices)
        holdings_df['current_value'] = holdings_df['amount'] / holdings_df['purchase_price'] * holdings_df['current_price']
        holdings_df['pct_change'] = (holdings_df['current_price'] / holdings_df['purchase_price'] - 1) * 100
        
        # Display holdings
        with st.container():
            st.markdown('<div class="portfolio-card">', unsafe_allow_html=True)
            st.subheader("Current Holdings")
            st.dataframe(
                holdings_df.style.format({
                    'amount': '₹{:.2f}',
                    'purchase_price': '₹{:.2f}',
                    'current_price': '₹{:.2f}',
                    'current_value': '₹{:.2f}',
                    'pct_change': '{:.2f}%'
                }).applymap(lambda x: 'color: #00B050' if x > 0 else 'color: #FF0000', 
                          subset=['pct_change']),
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # Portfolio optimization
        with st.container():
            st.markdown('<div class="portfolio-card">', unsafe_allow_html=True)
            st.subheader("⚡ Portfolio Optimization")
            
            try:
                tickers = holdings_df['ticker'].unique().tolist()
                data = get_historical_data(tickers)
                
                if data.empty or len(data) < 100:
                    st.error("Insufficient data for optimization (need at least 100 trading days)")
                    return
                
                # Calculate expected returns and covariance matrix
                mu = expected_returns.mean_historical_return(data)
                S = risk_models.sample_cov(data)
                
                # Handle risk level
                risk_level = st.session_state.risk_level
                ef = EfficientFrontier(mu, S)
                ef_minvol = EfficientFrontier(mu, S)
                ef_minvol.min_volatility()
                min_ret, min_vol, _ = ef_minvol.portfolio_performance()
                max_ret = mu.max()
                target_ret = min_ret + (max_ret - min_ret) * (risk_level - 1) / 9
                
                # Perform optimization
                ef.efficient_return(target_ret)
                weights = ef.clean_weights()
                
                # Create comparison dataframe
                current_total = holdings_df['current_value'].sum()
                current_weights = holdings_df.groupby('ticker')['current_value'].sum() / current_total
                optimized_weights = pd.Series(weights)
                
                comparison_df = pd.DataFrame({
                    'Current Weight': current_weights,
                    'Optimized Weight': optimized_weights
                }).dropna()
                comparison_df['Difference'] = comparison_df['Optimized Weight'] - comparison_df['Current Weight']
                
                # Display results
                st.write("Current vs. Optimized Allocation:")
                st.dataframe(
                    comparison_df.style.format({
                        'Current Weight': '{:.2%}',
                        'Optimized Weight': '{:.2%}',
                        'Difference': '{:+.2%}'
                    }).applymap(lambda x: 'color: #00B050' if x > 0 else 'color: #FF0000', 
                              subset=['Difference']),
                    use_container_width=True
                )
                
                # Show performance metrics
                st.subheader("📈 Performance Metrics")
                ret, vol, sharpe = ef.portfolio_performance()
                cols = st.columns(3)
                cols[0].metric("Expected Return", f"{ret:.2%}")
                cols[1].metric("Volatility", f"{vol:.2%}")
                cols[2].metric("Sharpe Ratio", f"{sharpe:.2f}")
                
            except Exception as e:
                st.error(f"Optimization failed: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()