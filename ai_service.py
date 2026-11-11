# ai_service.py
import re
import time
from google import genai
from google.genai import types
from openai import OpenAI
import streamlit as st
from config import GEMINI_MODEL, SYSTEM_PROMPT
from data_fetcher import fetch_top_institutional_stocks

GEMINI_FALLBACK_MODELS = [
    GEMINI_MODEL,
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.1-flash-lite",
]

GROQ_FALLBACK_MODELS = [
    "qwen-qwq-32b",
    "llama-3.3-70b-versatile",
]

BATCH_PROMPT = """你是一位專業股票技術分析師。請根據以下提供的盤後真實數據進行客觀、保守的獨立評估。
【嚴格格式規範】：
1. 絕對不可輸出「一、基本資料」或「五、最終報告」等任何標題、前言、問候語或條列式說明。
2. 必須「嚴格只輸出單一行」，用直線符號 `|` 分隔以下 6 個欄位：
總分/100 | 操作判斷(強烈看多/偏多/中性觀望/偏空/避免進場五選一) | 最佳掛單價 | 停損價 | 第一目標價 | 核心技術面與籌碼理由(30字內)

【正確輸出範例】：
68/100 | 偏多 | 135.5元 | 127.0元 | 148.0元 | 帶量突破季線且外資大買，但短線KD過熱宜拉回布局。

請嚴格依照上述單一行格式，分析以下股票數據：
"""

def _call_groq_fallback(prompt, temperature=0.1):
    """當 Gemini 失敗或未設定時，自動呼叫 Groq 免費備援模型"""
    groq_key = st.secrets.get("GROQ_API_KEY", "").strip()
    if not groq_key:
        return None

    groq_client = OpenAI(
        api_key=groq_key,
        base_url="https://groq.com",
        timeout=45.0,
    )

    for model_name in GROQ_FALLBACK_MODELS:
        try:
            st.toast(f"🚀 切換至 Groq 免費備援模型 ({model_name})...", icon="⚡")
            resp = groq_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            if resp.choices and resp.choices.message.content:
                text = resp.choices.message.content.strip()
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                return text
        except Exception:
            time.sleep(1)
            continue
    return None

def run_gemini_analysis(api_key, stock_data_text, is_batch=False):
    if is_batch:
        prompt = BATCH_PROMPT + stock_data_text
    else:
        prompt = SYSTEM_PROMPT + "\n\n請根據以下真實盤後數據進行完整分析：\n" + stock_data_text

    last_error = None
    if api_key:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=45000),
        )
        gen_config = types.GenerateContentConfig(
            temperature=0.1,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

        for model_name in dict.fromkeys(GEMINI_FALLBACK_MODELS):
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
                    st.toast("⚠️ 免費版每分鐘額度已滿，切換備援模型...", icon="🛑")
                else:
                    st.toast(f"⚠️ {model_name} 尖峰忙碌，自動切換備援模型...", icon="🔄")
                time.sleep(1)
                continue

    # Gemini 全數失敗或未設定 Key 時，自動啟用 Groq 備援
    groq_result = _call_groq_fallback(prompt, temperature=0.1)
    if groq_result:
        return groq_result

    raise RuntimeError(f"AI 暫時無法回應，請稍候 30 秒再試。（詳細原因：{last_error}）")

def fetch_ai_top10_picks(api_key):
    """結合 FinMind 真實三大法人買超數據與短線波段選股策略，產出 Markdown 表格與 10 檔代號"""
    finmind_token = st.secrets.get("FINMIND_TOKEN", "").strip()
    inst_data = fetch_top_institutional_stocks(token=finmind_token, top_n=15)

    if not inst_data:
        raise RuntimeError("無法從 FinMind 取得最新法人買超排行，請稍後再試。")

    prompt = f"""請你扮演一位專精於台股籌碼分析與短線波段交易的資深分析師。我需要你根據以下台灣股市最新交易日的真實法人買超前 15 名數據，精選出 10 檔具備短線爆發力且籌碼穩定的台股潛力個股（請排除 0050、0056 等純被動型 ETF）：

【最新交易日三大法人真實買超數據】：
{inst_data}

一、 篩選核心條件（需同時符合多項）：
1. 籌碼面：近 3 至 5 日「投信」或「外資」呈現連續買超，或剛由賣轉買且買超張數明顯放大（優先參考上方提供的真實買超張數）。
2. 技術面：股價處於多頭架構或剛帶量突破盤整區間，站穩短均線之上。
3. 位階與乖離：股價處於起漲階段，距離 5 日線的乖離率不宜過大（避免已連續漲停或暴漲超過 20% 的過熱標的）。
4. 題材催化：具備明確的營收成長動能、產業利多，或屬於當前市場主流資金聚集的族群。

二、 風險排除與嚴格客觀原則：
• 排除成交量過小（日均量低於 1,000 張）的冷門股，以及隔日沖風險過高之標的。
• 重要原則：買賣超張數必須嚴格引用上方提供的真實數據，未提供具體股價點位時，請以「5日線支撐」、「前波整理區間低點」、「量縮回測不破均線」等具體技術面條件給予短線操作建議，嚴禁憑空捏造假價格數字。

三、 輸出格式要求（請嚴格遵守）：
1. 請用標準 Markdown 表格呈現這 10 檔股票，欄位必須包含：

| 股票代號與名稱 | 所屬題材/族群 | 近期法人買賣超亮點 | 技術面進場優勢 | 短線操作建議 |
2. 表格下方最後一行，必須單獨用小括號列出這 10 檔的 4 位數股票代號，以逗號分隔，供系統一鍵帶入掃描器，例如：
(2330,2454,3661,3443,2408,1815,1802,1303,6223,2317)
"""

    full_text = None
    last_error = None

    if api_key:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=60000),
        )
        gen_config = types.GenerateContentConfig(
            temperature=0.2,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

        for model_name in dict.fromkeys(GEMINI_FALLBACK_MODELS):
            try:
                st.toast(f"🔍 正在透過 {model_name} 執行短線波段選股分析...", icon="🔥")
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=gen_config,
                )
                if response and response.text:
                    full_text = response.text.strip()
                    break
            except Exception as e:
                last_error = e
                st.toast(f"⚠️ {model_name} 暫時忙碌，切換備援模型...", icon="🔄")
                time.sleep(1)
                continue

    if not full_text:
        full_text = _call_groq_fallback(prompt, temperature=0.2)

    if full_text:
        match = re.search(r"\(([\d,\s]+)\)", full_text)
        if match:
            raw_codes = [c.strip() for c in match.group(1).split(",") if re.fullmatch(r"\d{4}", c.strip())]
        else:
            raw_codes = re.findall(r"\b\d{4}\b", full_text)
        codes = list(dict.fromkeys(raw_codes))[:10]
        return full_text, codes

    raise RuntimeError(f"AI 精選分析暫時無法回應，請稍後再試。（詳細原因：{last_error}）")
