# ai_service.py
import re
import time
from google import genai
from google.genai import types
import streamlit as st
from config import GEMINI_MODEL, SYSTEM_PROMPT

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

PICK_10_PROMPT = """你是一位專業台股分析師。請使用 Google 搜尋查詢最新台股三大法人買賣超動向與市場具備漲價、報價回升或強勢題材的熱門股，精選出 10 檔台股標的。
請依照以下格式輸出：
1. 先用簡短條列式列出這 10 檔股票的「代號 名稱」與「入選理由（法人買超或漲價題材）」。
2. 最後一行必須嚴格用半形括號與逗號列出純 4 位數股票代號，格式範例：(2330,2454,3661,3443,2408,1815,1802,1303,6223,2317)
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
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=45000),
    )

    gen_config = types.GenerateContentConfig(
        temperature=0.3,
        tools=[{"google_search": {}}],
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    last_error = None
    for model_name in dict.fromkeys(FALLBACK_MODELS):
        try:
            st.toast(f"🔍 正在透過 {model_name} 連網搜尋最新法人買超與潛力股...", icon="🌐")
            response = client.models.generate_content(
                model=model_name,
                contents=PICK_10_PROMPT,
                config=gen_config,
            )
            if response and response.text:
                text = response.text.strip()
                match = re.search(r"\(([0-9,\s]{10,})\)", text)
                if match:
                    raw_codes = [c.strip() for c in match.group(1).split(",") if c.strip()]
                else:
                    raw_codes = list(dict.fromkeys(re.findall(r"\b\d{4}\b", text)))
                codes = list(dict.fromkeys(raw_codes))[:10]
                return text, codes
        except Exception as e:
            last_error = e
            st.toast(f"⚠️ {model_name} 連網搜尋忙碌，切換備援模型...", icon="🔄")
            time.sleep(1)
            continue

    raise RuntimeError(f"AI 連網精選暫時無法回應，請稍後再試。（詳細原因：{last_error}）")
