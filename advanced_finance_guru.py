# advanced_finance_guru.py
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import google.generativeai as genai
from fpdf import FPDF
from bs4 import BeautifulSoup

class FinancialGuru:
    def __init__(self):
        self.risk_profile = self._calculate_risk_profile()
        self.nifty50 = self._get_index_data('^NSEI')
        self.risk_free_rate = 0.07
        self.tax_slabs = {
            'IND': [(250000, 0), (500000, 0.05),
                   (1000000, 0.2), (float('inf'), 0.3)]
        }
        self.gold_price = self._get_gold_price()
        self.btc_price = self._get_crypto_price('bitcoin')
        self.user_goals = {}

    def _calculate_risk_profile(self):
        """Dynamic risk assessment"""
        try:
            if not self.nifty50.empty:
                volatility = self.nifty50['Close'].pct_change().std() * np.sqrt(252)
                return min(int(10 - (volatility * 100)), 10)
            return 5
        except:
            return 5

    def _get_index_data(self, ticker):
        """Fetch index data from Yahoo Finance"""
        try:
            data = yf.download(ticker, period='1d', interval='5m')
            return data[['Close']].ffill().bfill() if not data.empty else pd.DataFrame()
        except Exception as e:
            st.error(f"Market Data Error: {str(e)}")
            return pd.DataFrame()

    def _get_gold_price(self):
        """Fetch live gold prices"""
        try:
            url = "https://www.goldpriceindia.com/"
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            price_div = soup.find("div", {"id": "currentPrice"})
            return float(price_div.text.strip().replace(',', '').split()[0])
        except Exception as e:
            st.error(f"Gold Price Error: {str(e)}")
            return np.nan

    def _get_crypto_price(self, coin_id):
        """Fetch cryptocurrency prices"""
        try:
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=inr"
            response = requests.get(url, timeout=10)
            return response.json()[coin_id]['inr']
        except Exception as e:
            st.error(f"Crypto Error: {str(e)}")
            return np.nan

    def calculate_sip(self, target, years, rate):
        """SIP Calculator"""
        try:
            monthly_rate = rate / 1200
            months = years * 12
            return (target * monthly_rate) / ((1 + monthly_rate)**months - 1)
        except Exception as e:
            st.error(f"SIP Error: {str(e)}")
            return 0

    def calculate_swp(self, corpus, years, rate):
        """SWP Calculator"""
        try:
            monthly_rate = rate / 1200
            months = years * 12
            return (corpus * monthly_rate) / (1 - (1 + monthly_rate)**-months)
        except Exception as e:
            st.error(f"SWP Error: {str(e)}")
            return 0

    def calculate_tax(self, income, deductions=0):
        """Tax Calculator"""
        try:
            taxable = max(income - deductions, 0)
            tax = 0
            remaining = taxable
            for slab, rate in self.tax_slabs['IND']:
                if remaining <= 0: break
                amount = min(remaining, slab)
                tax += amount * rate
                remaining -= amount
            return tax
        except Exception as e:
            st.error(f"Tax Error: {str(e)}")
            return 0

    def tax_saving_options(self, income):
        """Tax Saving Suggestions"""
        suggestions = []
        try:
            if income > 1500000:
                suggestions.append("Invest ₹1.5L in ELSS (Section 80C)")
            if income > 700000:
                suggestions.append("Health Insurance (Section 80D)")
            if income > 500000:
                suggestions.append("NPS Investment (Section 80CCD)")
        except Exception as e:
            st.error(f"Tax Suggestion Error: {str(e)}")
        return suggestions

    def calculate_emi(self, principal, rate, years):
        """EMI Calculator"""
        try:
            monthly_rate = rate / 1200
            months = years * 12
            return (principal * monthly_rate * (1 + monthly_rate)**months) / \
                   ((1 + monthly_rate)**months - 1)
        except Exception as e:
            st.error(f"EMI Error: {str(e)}")
            return 0

    def create_goal_plan(self, goal_type, current_cost, years, inflation=6):
        """Goal Planning System"""
        try:
            future_value = current_cost * (1 + inflation/100)**years
            sip = self.calculate_sip(future_value, years, 12)
            swp = self.calculate_swp(future_value, 10, 8)
            return {
                'future_value': future_value,
                'monthly_sip': sip,
                'monthly_swp': swp
            }
        except Exception as e:
            st.error(f"Goal Planning Error: {str(e)}")
            return {}

    def ai_advisor(self, prompt):
        """AI Financial Advisor"""
        try:
            genai.configure(api_key=st.secrets["GEMINI_KEY"])
            model = genai.GenerativeModel('gemini-pro')
            
            market_status = (
                f"Nifty 50: {self._safe_nifty_value()}\n"
                f"Gold: ₹{self.gold_price:,.2f}/10g\n"
                f"Bitcoin: ₹{self.btc_price:,.2f}\n"
                f"Risk-Free Rate: {self.risk_free_rate*100}%"
            )
            
            response = model.generate_content(
                f"Act as SEBI-certified financial advisor. Current market:\n{market_status}\n"
                f"Question: {prompt}\n"
                "Answer in markdown with bullet points:"
            )
            return response.text
        except Exception as e:
            return f"AI Service Error: {str(e)}"

    def _safe_nifty_value(self):
        """Safe Nifty Value Formatter"""
        try:
            return f"₹{self.nifty50['Close'].iloc[-1]:,.2f}" if not self.nifty50.empty else "N/A"
        except:
            return "N/A"

    def generate_report(self, data):
        """PDF Report Generator"""
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Financial Report", ln=1, align='C')
            for k, v in data.items():
                pdf.cell(200, 10, txt=f"{k}: {v}", ln=1)
            return pdf.output(dest='S').encode('latin1')
        except Exception as e:
            st.error(f"Report Error: {str(e)}")
            return b''

def main():
    st.set_page_config(page_title="AI Finance Guru Pro", layout="wide")
    st.title("💰 AI Finance Guru - Complete Wealth Management")
    
    guru = FinancialGuru()
    
    # Real-time Dashboard
    with st.container():
        cols = st.columns(4)
        cols[0].metric("Nifty 50", guru._safe_nifty_value(), help="Live Nifty 50 Index Value")
        gold_value = f"₹{guru.gold_price:,.2f}/10g" if not np.isnan(guru.gold_price) else "N/A"
        cols[1].metric("Gold (10g)", gold_value)
        cols[2].metric("Bitcoin", f"₹{guru.btc_price:,.2f}" if not np.isnan(guru.btc_price) else "N/A")
        cols[3].metric("Risk Profile", f"{guru.risk_profile}/10")

    # Main Interface
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["🏡 Goal Planner", "📈 SIP Calculator", "📉 SWP Calculator",
         "💰 Tax Planner", "🏦 Loan Calculator", "🤖 AI Advisor"]
    )

    with tab1:
        st.subheader("Goal-Based Financial Planning")
        goal_type = st.selectbox("Select Goal Type",
                               ["Education", "Retirement", "Home", "Travel"])
        current_cost = st.number_input("Current Cost (₹)", value=1000000)
        years = st.slider("Years Until Goal", 1, 30, 10)
        inflation = st.slider("Expected Inflation (%)", 3.0, 15.0, 6.0)
        
        if st.button("Generate Plan"):
            plan = guru.create_goal_plan(goal_type, current_cost, years, inflation)
            st.metric("Future Value", f"₹{plan['future_value']:,.0f}")
            st.metric("Monthly SIP Needed", f"₹{plan['monthly_sip']:,.0f}")
            st.metric("Post-Goal SWP", f"₹{plan['monthly_swp']:,.0f}/month")

    with tab2:
        st.subheader("SIP Calculator")
        target = st.number_input("Financial Goal (₹)", value=5000000)
        years = st.slider("Investment Period (years)", 1, 30, 10)
        rate = st.slider("Expected Returns (%)", 5.0, 20.0, 12.0)
        sip = guru.calculate_sip(target, years, rate)
        st.metric("Monthly SIP Required", f"₹{sip:,.0f}")

    with tab3:
        st.subheader("SWP Calculator")
        corpus = st.number_input("Current Corpus (₹)", value=5000000)
        years = st.slider("Withdrawal Period (years)", 1, 30, 10)
        rate = st.slider("Expected Growth (%)", 5.0, 15.0, 8.0)
        swp = guru.calculate_swp(corpus, years, rate)
        st.metric("Monthly Withdrawal", f"₹{swp:,.0f}")

    with tab4:
        st.subheader("Tax Planning")
        income = st.number_input("Annual Income (₹)", value=1500000)
        deductions = st.number_input("Deductions (₹)", value=150000)
        tax = guru.calculate_tax(income, deductions)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Tax Liability", f"₹{tax:,.0f}")
        with col2:
            st.write("### Tax Saving Options")
            suggestions = guru.tax_saving_options(income)
            for suggestion in suggestions:
                st.write(f"- {suggestion}")

    with tab5:
        st.subheader("Loan Calculator")
        principal = st.number_input("Loan Amount (₹)", value=5000000)
        rate = st.slider("Interest Rate (%)", 5.0, 20.0, 10.5)
        tenure = st.slider("Loan Tenure (years)", 1, 30, 15)
        emi = guru.calculate_emi(principal, rate, tenure)
        st.metric("Monthly EMI", f"₹{emi:,.0f}")

    with tab6:
        st.subheader("AI Financial Advisor")
        query = st.text_area("Ask Financial Question", height=100)
        if st.button("Get AI Advice"):
            with st.spinner("Analyzing..."):
                response = guru.ai_advisor(query)
                st.markdown(response)
        
        if st.button("Generate Full Report"):
            report_data = {
                "Financial Goal": f"₹{current_cost:,.0f} in {years} years",
                "Monthly SIP": f"₹{sip:,.0f}",
                "Monthly SWP": f"₹{swp:,.0f}",
                "Tax Liability": f"₹{tax:,.0f}",
                "Loan EMI": f"₹{emi:,.0f}",
                "Risk Profile": f"{guru.risk_profile}/10"
            }
            pdf = guru.generate_report(report_data)
            st.download_button("Download PDF Report", pdf, "financial_report.pdf")

if __name__ == "__main__":
    main()