# ai_service.py
import time
from google import genai
from config import GEMINI_MODEL, SYSTEM_PROMPT

# 備援模型清單（主模型塞車時自動依序切換）
FALLBACK_MODELS = [GEMINI_MODEL, "gemini-2.5-flash-lite", "gemini-2.0-flash"]

def run_gemini_analysis_stream(api_key, stock_data_text, is_batch=False):
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

    last_error = None
    for model_name in dict.fromkeys(FALLBACK_MODELS):
        for attempt in range(2):
            try:
                response = client.models.generate_content_stream(
                    model=model_name, contents=prompt
                )
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
                return
            except Exception as e:
                last_error = e
                err_str = str(e)
                # 遇到 503 塞車或 429 頻率限制時，稍等 2 秒重試或換下一個備援模型
                if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str:
                    time.sleep(2)
                    continue
                raise e

    raise last_error
