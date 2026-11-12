# app.py
import json
import mimetypes
import os
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ai_service import fetch_ai_top10_picks, run_gemini_analysis
from data_fetcher import fetch_stock_package, get_stock_name_map

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

st.set_page_config(page_title="台股 AI 100分獨立評分系統", layout="wide")

# --- 修正 st.metric 漲跌顏色為台股習慣（紅漲綠跌） ---
st.markdown(
    """
    <style>
    [data-testid="stMetricDelta"]:has(svg[data-testid="stMetricDeltaIcon-Up"]) {
        color: #ef5350 !important;
        background-color: rgba(239, 83, 80, 0.12) !important;
    }
    [data-testid="stMetricDelta"]:has(svg[data-testid="stMetricDeltaIcon-Down"]) {
        color: #26a69a !important;
        background-color: rgba(38, 166, 154, 0.12) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 台股全維度技術分析與 AI 100分獨立評分系統")

gemini_key = st.secrets.get("GEMINI_API_KEY", "").strip()
groq_key = st.secrets.get("GROQ_API_KEY", "").strip()
finmind_token = st.secrets.get("FINMIND_TOKEN", "").strip()

# --- 自選股本地端永久儲存管理 (watchlist.json) ---
WATCHLIST_FILE = "watchlist.json"
DEFAULT_WATCHLIST = {
    "2330": "台積電",
    "2382": "廣達",
    "2317": "鴻海",
    "2454": "聯發科",
    "3231": "緯創",
    "2603": "長榮",
    "2881": "富邦金",
    "0050": "元大台灣50",
}


def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and data:
                    return data
        except Exception:
            pass
    save_watchlist(DEFAULT_WATCHLIST)
    return DEFAULT_WATCHLIST.copy()


def save_watchlist(watchlist_dict):
    with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
        json.dump(watchlist_dict, f, ensure_ascii=False, indent=2)


watchlist = load_watchlist()

# 初始化批量掃描輸入框狀態
if "batch_input_str" not in st.session_state:
    st.session_state["batch_input_str"] = ", ".join(list(watchlist.keys())[:10])
if "ai_top10_report" not in st.session_state:
    st.session_state["ai_top10_report"] = ""
if "ai_top10_codes" not in st.session_state:
    st.session_state["ai_top10_codes"] = []

# --- 側邊欄：選股與自選股增刪管理 ---
with st.sidebar:
    st.header("📌 單股深度分析設定")

    options = [f"{sid} {name}".strip() for sid, name in watchlist.items()]
    options.insert(0, "✏️ 手動輸入其他代號")

    selected_label = st.selectbox("常用自選股快捷選單", options)
    if selected_label == "✏️ 手動輸入其他代號":
        stock_id = st.text_input("請輸入股票代號 (例如 3711)", value="3711").strip()
    else:
        stock_id = selected_label.split()[0]

    days_back = st.slider("回溯天數", 250, 730, 250)

    st.divider()
    with st.expander("⚙️ 管理常用自選股清單（永久保留）", expanded=False):
        new_sid = st.text_input("輸入欲新增的代號", placeholder="例如 3711").strip()
        if st.button("➕ 加入自選股", use_container_width=True):
            if not new_sid:
                st.warning("請先輸入股票代號！")
            elif new_sid in watchlist:
                st.info(f"{new_sid} 已經在清單中了。")
            else:
                name_map = get_stock_name_map(finmind_token)
                stock_name = name_map.get(new_sid, "")
                if not stock_name:
                    st.error("查無此股票代號，請確認是否正確。")
                else:
                    watchlist[new_sid] = stock_name
                    save_watchlist(watchlist)
                    st.success(f"已新增 {new_sid} {stock_name}！")
                    time.sleep(0.5)
                    st.rerun()

        if len(watchlist) > 1:
            del_options = [f"{sid} {name}".strip() for sid, name in watchlist.items()]
            del_target = st.selectbox("選擇欲移除的股票", del_options)
            if st.button("🗑️ 移除選取股票", use_container_width=True):
                del_sid = del_target.split()[0]
                watchlist.pop(del_sid, None)
                save_watchlist(watchlist)
                st.success(f"已移除 {del_target}！")
                time.sleep(0.5)
                st.rerun()

main_tab1, main_tab2 = st.tabs(["🎯 單股看盤與 AI 深度評分", "📋 自選股清單批量掃描"])

with main_tab1:
    pkg = fetch_stock_package(stock_id, days_back, finmind_token)
    if pkg is None:
        st.warning("找不到該股票代號或資料不足，請確認代號是否正確。")
    else:
        df_d = pkg["day"]
        latest, prev = df_d.iloc[-1], df_d.iloc[-2]
        chg = latest["Close"] - prev["Close"]
        chg_pct = (chg / prev["Close"]) * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{pkg['title']} 最新收盤", f"{latest['Close']:.2f}", f"{chg:+.2f} ({chg_pct:+.2f}%)")
        c2.metric("成交量 (張)", f"{int(latest['Volume_Lot']):,}")
        c3.metric("日線 9K / 9D", f"{latest['KD_K']:.1f} / {latest['KD_D']:.1f}")
        c4.metric("日線 5RSI / 10RSI", f"{latest['RSI_5']:.1f} / {latest['RSI_10']:.1f}")

        # --- 三層子圖：1.日K線 2.成交量+5日均量線 3.三大法人合計買賣超 ---
        plot_df = df_d.tail(90)
        vol_colors = ["#ef5350" if c >= o else "#26a69a" for c, o in zip(plot_df["Close"], plot_df["Open"])]
        inst_colors = ["#ef5350" if v >= 0 else "#26a69a" for v in plot_df["Inst_Net"]]

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.56, 0.22, 0.22],
        )

        # 第 1 層：日K與均線
        fig.add_trace(
            go.Candlestick(
                x=plot_df.index,
                open=plot_df["Open"],
                high=plot_df["High"],
                low=plot_df["Low"],
                close=plot_df["Close"],
                increasing_line_color="#ef5350",
                decreasing_line_color="#26a69a",
                name="日K",
            ),
            row=1, col=1,
        )
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["MA5"], line=dict(color="orange", width=1.2), name="5MA"), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["MA20"], line=dict(color="cyan", width=1.2), name="20MA"), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["MA60"], line=dict(color="magenta", width=1.2), name="60MA"), row=1, col=1)

        # 第 2 層：成交量柱狀圖 + 5日均量線
        fig.add_trace(go.Bar(x=plot_df.index, y=plot_df["Volume_Lot"], marker_color=vol_colors, name="成交量(張)"), row=2, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Vol_MA5"], line=dict(color="#ffd54f", width=1.5), name="5日均量"), row=2, col=1)

        # 第 3 層：三大法人合計買賣超
        fig.add_trace(go.Bar(x=plot_df.index, y=plot_df["Inst_Net"], marker_color=inst_colors, name="三大法人(張)"), row=3, col=1)

        fig.update_layout(
            height=680,
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        fig.update_yaxes(title_text="價格", row=1, col=1)
        fig.update_yaxes(title_text="成交量", row=2, col=1)
        fig.update_yaxes(title_text="法人買賣超", row=3, col=1)

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🤖 AI 100分獨立評分報告")
        if st.button("🚀 立即執行 AI 深度評分", type="primary"):
            if not gemini_key and not groq_key:
                st.error("請先在 Secrets 中設定 GEMINI_API_KEY 或 GROQ_API_KEY！")
            else:
                with st.spinner("⚡ AI 正在快速運算評分中（約 3～8 秒）..."):
                    try:
                        report = run_gemini_analysis(gemini_key, pkg["data_text"], is_batch=False)
                        st.markdown(report)
                    except Exception as e:
                        st.error(f"AI 分析發生錯誤：{e}")

        with st.expander("檢視傳送給 AI 的原始盤後數據包"):
            st.code(pkg["data_text"], language="text")

with main_tab2:
    # --- 區塊一：AI 盤後精選 10 檔潛力股 ---
    st.subheader("🔥 AI 盤後精選 10 檔潛力股（即時連網籌碼與題材）")
    col_pick1, col_pick2 = st.columns([1, 1])
    with col_pick1:
        if st.button("✨ AI 自動搜尋並推薦 10 檔潛力股", use_container_width=True):
            if not gemini_key:
                st.error("請先在 .streamlit/secrets.toml 中設定 GEMINI_API_KEY！")
            else:
                with st.spinner("🌐 AI 正在連網搜尋最新三大法人買超與漲價題材股..."):
                    try:
                        report_text, codes = fetch_ai_top10_picks(gemini_key)
                        st.session_state["ai_top10_report"] = report_text
                        st.session_state["ai_top10_codes"] = codes
                    except Exception as e:
                        st.error(f"AI 精選搜尋失敗：{e}")

    if st.session_state["ai_top10_report"]:
        with st.expander("📋 查看 AI 精選 10 檔推薦名單與理由", expanded=True):
            st.markdown(st.session_state["ai_top10_report"])
            if st.session_state["ai_top10_codes"]:
                codes_preview = ", ".join(st.session_state["ai_top10_codes"])
                st.info(f"偵測到的 10 檔代號：`{codes_preview}`")
                if st.button("📥 一鍵帶入下方批量掃描器", type="primary"):
                    st.session_state["batch_input_str"] = codes_preview
                    st.rerun()

    st.divider()

    # --- 區塊二：自選股多檔批量掃描器 ---
    st.subheader("📋 自選股多檔批量掃描器（單次上限 10 檔）")
    col_reset1, col_reset2 = st.columns([3, 1])
    with col_reset2:
        if st.button("🔄 恢復為我的常用自選股", use_container_width=True):
            st.session_state["batch_input_str"] = ", ".join(list(watchlist.keys())[:10])
            st.rerun()

    watchlist_input = st.text_input(
        "輸入自選股代號（請用逗號分隔，單次最多 10 檔以維持最佳速度與免費額度）",
        key="batch_input_str",
    )

    if st.button("🔍 開始批量掃描自選股", type="primary"):
        if not gemini_key:
            st.error("請先在 .streamlit/secrets.toml 中設定 GEMINI_API_KEY！")
        else:
            stock_list = list(dict.fromkeys([s.strip() for s in watchlist_input.split(",") if s.strip()]))

            if not stock_list:
                st.warning("請至少輸入 1 檔股票代號！")
            elif len(stock_list) > 10:
                st.error(f"⚠️ 您目前輸入了 **{len(stock_list)}** 檔股票，單次批量掃描上限為 **10 檔**。請刪減部分代號後再執行，以免觸發免費版 API 速率限制。")
            else:
                progress_bar = st.progress(0)
                results_container = st.container()

                for idx, sid in enumerate(stock_list):
                    with st.spinner(f"正在抓取並評分 {sid} ({idx+1}/{len(stock_list)})..."):
                        pkg = fetch_stock_package(sid, days_back, finmind_token)
                        if pkg is None:
                            results_container.warning(f"❌ **{sid}**：查無資料或代號錯誤。")
                        else:
                            try:
                                summary_res = run_gemini_analysis(gemini_key, pkg["data_text"], is_batch=True)
                                with results_container.expander(f"📌 {pkg['title']} 掃描結果（收盤 {pkg['day'].iloc[-1]['Close']:.2f} 元）", expanded=True):
                                    st.markdown(summary_res)
                            except Exception as e:
                                results_container.error(f"❌ **{sid}** 分析失敗：{e}")
                    progress_bar.progress((idx + 1) / len(stock_list))
                    time.sleep(1.5)
                st.success("✅ 所有自選股批量掃描完成！")
