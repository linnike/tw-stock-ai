# ai_service.py
import time
from google import genai
from google.genai import types
import streamlit as st
from config import GEMINI_MODEL, SYSTEM_PROMPT

FALLBACK_MODELS = [GEMINI_MODEL, "gemini-3.5-flash-lite", "gemini-3.6-flash"]

# 獨立的批量掃描專用指令（徹底避免與五大段完整報告規則衝突）
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
