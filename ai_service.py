# ai_service.py
import time
from google import genai
from google.genai import types
import streamlit as st
from config import GEMINI_MODEL, SYSTEM_PROMPT

# 免費版額度最寬鬆、速度最快的兩個模型
FALLBACK_MODELS = [GEMINI_MODEL, "gemini-2.5-flash-lite"]

def run_gemini_analysis(api_key, stock_data_text, is_batch=False):
    client = genai.Client(api_key=api_key)

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

    # 關閉深度思考延遲，讓免費版模型直接用最高速生成
    gen_config = types.GenerateContentConfig(
        temperature=0.2,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    last_error = None
    for model_name in dict.fromkeys(FALLBACK_MODELS):
        try:
            st.toast(f"🤖 正在呼叫 {model_name} 生成報告...", icon="⚡")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=gen_config,
            )
            if response and response.text:
                return response.text
        except Exception as e:
            last_error = e
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                st.toast("⚠️ 免費版每分鐘請求次數已滿，請稍等 30 秒再試！", icon="🛑")
            else:
                st.toast(f"⚠️ {model_name} 暫時忙碌，切換備援模型...", icon="🔄")
            time.sleep(2)
            continue

    raise RuntimeError(f"AI 暫時無法回應，請稍候 30 秒再試。（詳細原因：{last_error}）")
