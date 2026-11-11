# ai_service.py
import time
from google import genai
from google.genai import types
import streamlit as st
from config import GEMINI_MODEL, SYSTEM_PROMPT

# 現行穩定且快速的備援模型清單
FALLBACK_MODELS = [GEMINI_MODEL, "gemini-2.5-flash-lite", "gemini-2.5-pro"]

def run_gemini_analysis_stream(api_key, stock_data_text, is_batch=False):
    # 設定 15 秒連線逾時，避免伺服器無回應時無限期卡死
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=15.0),
    )

    if is_batch:
        prompt = (
            SYSTEM_PROMPT
            + "\n\n【重要附加指令】：因為這是多股批量掃描，請直接精簡輸出「五、最終報告」的內容，並用一行格式總結：總分 | 操作判斷 | 最佳掛單價 | 停損價 | 第一目標價 | 核心理由，不要輸出冗長過程。\n\n數據如下：\n"
            + stock_data_text
        )
    else:
        prompt = (
            SYSTEM_PROMPT + "\n\n請根據以下真實盤後數據進行完整分析：\n" + stock_data_text
        )

    last_error = None
    for model_name in dict.fromkeys(FALLBACK_MODELS):
        try:
            st.toast(f"🤖 正在連線 {model_name} 進行分析...", icon="⏳")
            response = client.models.generate_content_stream(
                model=model_name, contents=prompt
            )
            has_output = False
            for chunk in response:
                if chunk.text:
                    has_output = True
                    yield chunk.text
            if has_output:
                return
        except Exception as e:
            last_error = e
            st.toast(f"⚠️ {model_name} 忙碌或逾時，自動切換備援模型...", icon="🔄")
            time.sleep(1)
            continue

    raise RuntimeError(f"所有 AI 模型目前皆無回應，請稍後再試。（最後錯誤：{last_error}）")
