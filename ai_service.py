# ai_service.py
from google import genai
from config import GEMINI_MODEL, SYSTEM_PROMPT

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

    response = client.models.generate_content(
        model=GEMINI_MODEL, contents=prompt
    )
    return response.text
