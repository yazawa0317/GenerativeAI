import fitz  # PyMuPDF を使用
import os
import random

from langchain.chains.qa_generation.prompt import CHAT_PROMPT as prompt
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.runnables.base import RunnableEach
from openai import AzureOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1️⃣ PDF からテキストを抽出する関数
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    return text

# 2️⃣ LLM を呼び出す関数
def call_openai(input_text, model):
    if not isinstance(input_text, str):
        input_text = str(input_text)

    endpoint = os.getenv("AZURE_ENDPOINT_URL")
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("OPENAI_API_VERSION")


    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version=api_version,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "あなたは親切なアシスタントです。与えられた入力から質問と回答のペアを作成します。なるべく具体的な数字が回答に含まれるようにしてください。必ず日本語にしてください。"},
            {"role": "user", "content": input_text}
        ]
    )
    return response.choices[0].message.content

# 3️⃣ メインの関数（Runnableを維持）
def generateQA(text, model='gpt-4o', chunk=512, overlap=128, num_questions=5):
    # ❶ テキスト分割
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        separators=["\n", "。", "、", " "],
        encoding_name='o200k_base',
        chunk_size=chunk,
        chunk_overlap=overlap,
    )

    def split_text_as_string(text):
        docs = text_splitter.create_documents([text])
        return [doc.page_content for doc in docs]

    def split_text_with_sampling(text):
        """RecursiveCharacterTextSplitter + ランダムサンプリングのミックス"""
        docs = text_splitter.create_documents([text])
        split_texts = [doc.page_content for doc in docs]

        # 🎯 ここでランダムに num_questions 個のチャンクを選択
        if num_questions < len(split_texts):
            selected_texts = random.sample(split_texts, num_questions)
        else:
            selected_texts = split_texts  # 十分な数がない場合は全て

        return selected_texts


#    split_text = RunnableLambda(split_text_as_string)
    split_text = RunnableLambda(split_text_with_sampling)

    # ❷ LLM 呼び出しを Runnable 化
    llm_runnable = RunnableLambda(lambda input_text: call_openai(input_text, model))

    # ❸ Runnable チェーンを定義
    chain = RunnableParallel(
        text=RunnablePassthrough(),
        questions=split_text | RunnableEach(bound=prompt | llm_runnable | JsonOutputParser())
    )

    # ❹ 実行して結果を取得
    result = chain.invoke(text)
    
    return result.get('questions', [])

# 4️⃣ 実行例
if __name__ == "__main__":

    from dotenv import load_dotenv
    # .envからAOAI接続ようのパラメータを読み込み環境変数にセット
    load_dotenv()

    pdf_text = extract_text_from_pdf("C:\\Develop\\python\\GenAI\\RAG\\Spliter\\doc\\test.pdf")
    results = generateQA(text=pdf_text, model="gpt-4o", chunk=1024, overlap=128, num_questions=10)

    for qa in results:
        print(qa)
