# ai_service.py
import re
import time
from google import genai
from google.genai import types
import streamlit as st
from config import GEMINI_MODEL, SYSTEM_PROMPT
from data_fetcher import fetch_top_institutional_stocks

FALLBACK_MODELS = [GEMINI_MODEL, "gemini-3.5-flash-lite", "gemini-3.6-flash"]

BATCH_PROMPT = """你是一位專業股票技術分析師。請根據以下提供的盤後真實數據進行客觀、保守的獨立評估。
【嚴格格式規範】：
1. 絕對不可輸出「一、基本資料」或「五、最終報告」等任何標題、前言、問候語或條列式說明。
2. 必須「嚴格只輸出單一行」，用直線符號 `|` 分隔以下 6 個欄位：
總分/100 | 操作判斷(強烈看多/偏多/中性觀望/偏空/避免進場五選一) | 最佳掛單價 | 停損價 | 第一目標價 | 核心技術面與籌碼理由(30字內)

【正確輸出範例】：
68/100 | 偏多 | 135.5元 | 127.0元 | 148.0元 | 帶量突破季線且外資大買，但短線KD過熱宜拉回布局。

請嚴格依照上述單一行格式，分析以下股票數據：
"""

def run_gemini_analysis(api_key, stock_data_text, is_batch=False):
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=35000),
    )

    if is_batch:
        prompt = BATCH_PROMPT + stock_data_text
    else:
        prompt = SYSTEM_PROMPT + "\n\n請根據以下真實盤後數據進行完整分析：\n" + stock_data_text

    gen_config = types.GenerateContentConfig(
        temperature=0.1,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    last_error = None
    for model_name in dict.fromkeys(FALLBACK_MODELS):
        try:
            st.toast(f"⚡ 正在使用 {model_name} 生成報告...", icon="🤖")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=gen_config,
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            last_error = e
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                st.toast("⚠️ 免費版每分鐘額度已滿，請等 30 秒再試！", icon="🛑")
            else:
                st.toast(f"⚠️ {model_name} 尖峰忙碌，自動切換備援模型...", icon="🔄")
            time.sleep(1)
            continue

    raise RuntimeError(f"AI 暫時無法回應，請稍候 30 秒再試。（詳細原因：{last_error}）")

def fetch_ai_top10_picks(api_key):
    """直接讀取 FinMind 真實三大法人買超數據，由 AI 精選 10 檔潛力股（完全不消耗連網額度）"""
    finmind_token = st.secrets.get("FINMIND_TOKEN", "").strip()
    inst_data = fetch_top_institutional_stocks(token=finmind_token, top_n=15)

    if not inst_data:
        raise RuntimeError("無法從 FinMind 取得最新法人買超排行，請稍後再試。")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=35000),
    )

    prompt = f"""你是一位專業台股分析師。以下是台灣股市最新交易日「三大法人合計買超前 15 名」的真實籌碼數據：

{inst_data}

請從上述真實籌碼名單中，排除純防禦型或波動極低的 ETF（如 0050、0056 等），精選出 10 檔最具成長動能與產業題材的台股個股。
請嚴格遵守以下輸出格式（不要有前言或結語）：
1. 先條列輸出 10 檔股票（格式：1. 代號 名稱：買超張數與產業題材簡評，每檔 25 字內）。
2. 在最後一行，必須單獨用小括號列出這 10 檔的 4 位數股票代號，以逗號分隔，例如：(2330,2454,3661,3443,2408,1815,1802,1303,6223,2317)
"""

    gen_config = types.GenerateContentConfig(
        temperature=0.2,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    last_error = None
    for model_name in dict.fromkeys(FALLBACK_MODELS):
        try:
            st.toast(f"🔍 正在透過 {model_name} 分析最新法人籌碼精選股...", icon="🔥")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=gen_config,
            )
            if response and response.text:
                full_text = response.text.strip()
                match = re.search(r"\(([\d,\s]+)\)", full_text)
                if match:
                    raw_codes = [c.strip() for c in match.group(1).split(",") if re.fullmatch(r"\d{4}", c.strip())]
                else:
                    raw_codes = re.findall(r"\b\d{4}\b", full_text)
                codes = list(dict.fromkeys(raw_codes))[:10]
                return full_text, codes
        except Exception as e:
            last_error = e
            st.toast(f"⚠️ {model_name} 暫時忙碌，切換備援模型...", icon="🔄")
            time.sleep(1)
            continue

    raise RuntimeError(f"AI 精選分析暫時無法回應，請稍後再試。（詳細原因：{last_error}）")
