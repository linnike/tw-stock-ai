# data_fetcher.py
from datetime import datetime, timedelta
from FinMind.data import DataLoader
import pandas as pd
import streamlit as st
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator

@st.cache_data(ttl=86400)
def get_stock_name_map(token=""):
    dl = DataLoader()
    if token:
        dl.login_by_token(api_token=token)
    df_info = dl.taiwan_stock_info()
    if df_info.empty:
        return {}
    return dict(zip(df_info["stock_id"], df_info["stock_name"]))

@st.cache_data(ttl=3600)
def fetch_stock_package(sid, days, token=""):
    name_map = get_stock_name_map(token)
    stock_name = name_map.get(sid, "")
    stock_title = f"{sid} {stock_name}".strip()
    dl = DataLoader()
    if token:
        dl.login_by_token(api_token=token)

    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    rev_start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")

    df_day = dl.taiwan_stock_daily(stock_id=sid, start_date=start_date)
    if df_day.empty or len(df_day) < 60:
        return None

    df_day["date"] = pd.to_datetime(df_day["date"])
    df_day = df_day.sort_values("date").set_index("date")
    df_day.rename(
        columns={"open": "Open", "max": "High", "min": "Low", "close": "Close", "Trading_Volume": "Volume"},
        inplace=True,
    )
    df_day["Volume_Lot"] = (df_day["Volume"] / 1000).round(0)
    df_day["Vol_MA5"] = df_day["Volume_Lot"].rolling(5).mean().round(0)

    agg_dict = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume_Lot": "sum"}
    df_week = df_day.resample("W-FRI").agg(agg_dict).dropna()
    df_month = df_day.resample("ME").agg(agg_dict).dropna()

    for df in (df_day, df_week, df_month):
        for ma in (5, 10, 20, 60):
            df[f"MA{ma}"] = SMAIndicator(close=df["Close"], window=ma).sma_indicator()
        df["RSI_5"] = RSIIndicator(close=df["Close"], window=5).rsi()
        df["RSI_10"] = RSIIndicator(close=df["Close"], window=10).rsi()

        macd_obj = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
        df["MACD_DIF"] = macd_obj.macd()
        df["MACD_SIGNAL"] = macd_obj.macd_signal()
        df["MACD_HIST"] = macd_obj.macd_diff()

        kd_obj = StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"], window=9, smooth_window=3)
        df["KD_K"] = kd_obj.stoch()
        df["KD_D"] = kd_obj.stoch_signal()

    df_inst = dl.taiwan_stock_institutional_investors(stock_id=sid, start_date=start_date)
    df_margin = dl.taiwan_stock_margin_purchase_short_sale(stock_id=sid, start_date=start_date)
    df_rev = dl.taiwan_stock_month_revenue(stock_id=sid, start_date=rev_start)

    latest, w_lat, m_lat = df_day.iloc[-1], df_week.iloc[-1], df_month.iloc[-1]

    inst_summary = "無資料"
    df_day["Inst_Net"] = 0.0
    if not df_inst.empty:
        df_inst["date"] = pd.to_datetime(df_inst["date"])
        df_inst["net"] = ((df_inst["buy"] - df_inst["sell"]) / 1000).round(0)
        inst_p = df_inst.pivot_table(index="date", columns="name", values="net", aggfunc="sum")
        df_day["Inst_Net"] = inst_p.sum(axis=1).reindex(df_day.index).fillna(0)
        inst_summary = inst_p.tail(10).to_string()

    margin_summary = "無資料"
    if not df_margin.empty:
        margin_summary = df_margin[["date", "MarginPurchaseTodayBalance", "ShortSaleTodayBalance"]].tail(10).to_string(index=False)

    rev_summary = "無資料"
    if not df_rev.empty:
        rev_summary = df_rev[["revenue_year", "revenue_month", "revenue"]].tail(3).to_string(index=False)

    data_text = f"""
【股票代號與名稱】：{stock_title}
【最新交易日】：{df_day.index[-1].strftime('%Y-%m-%d')}
【當日價量】：開 {latest['Open']} / 高 {latest['High']} / 低 {latest['Low']} / 收 {latest['Close']} / 成交量 {int(latest['Volume_Lot'])} 張 (5日均量 {int(df_day['Volume_Lot'].tail(5).mean())} 張, 10日均量 {int(df_day['Volume_Lot'].tail(10).mean())} 張)
【波段高低點】：近20日高 {df_day['High'].tail(20).max()} / 近20日低 {df_day['Low'].tail(20).min()} / 近60日高 {df_day['High'].tail(60).max()} / 近60日低 {df_day['Low'].tail(60).min()}
【日線指標】：5MA={latest['MA5']:.2f}, 10MA={latest['MA10']:.2f}, 20MA={latest['MA20']:.2f}, 60MA={latest['MA60']:.2f} | 9K={latest['KD_K']:.2f}, 9D={latest['KD_D']:.2f} | DIF={latest['MACD_DIF']:.2f}, MACD={latest['MACD_SIGNAL']:.2f} | 5RSI={latest['RSI_5']:.2f}, 10RSI={latest['RSI_10']:.2f}
【週線指標】：5T={w_lat['MA5']:.2f}, 10T={w_lat['MA10']:.2f}, 20T={w_lat['MA20']:.2f} | 9K={w_lat['KD_K']:.2f}, 9D={w_lat['KD_D']:.2f} | DIF={w_lat['MACD_DIF']:.2f}, MACD={w_lat['MACD_SIGNAL']:.2f} | 5RSI={w_lat['RSI_5']:.2f}
【月線指標】：20T={m_lat['MA20']:.2f} | DIF={m_lat['MACD_DIF']:.2f}, MACD={m_lat['MACD_SIGNAL']:.2f} | 5RSI={m_lat['RSI_5']:.2f}, 10RSI={m_lat['RSI_10']:.2f}
【近10日三大法人買賣超(張)】：\n{inst_summary}
【近10日融資融券餘額(張)】：\n{margin_summary}
【近3個月單月合併營收】：\n{rev_summary}
"""
    return {"title": stock_title, "day": df_day, "week": df_week, "month": df_month, "data_text": data_text}
