import os
import random

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.runnables.base import RunnableEach
from langchain_openai import AzureChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from textspliter import set_encoding_type
from text_utils import SPLIT_DOCS_PROMPT as prompt


# LLMの初期化
def set_llm(model='gpt-4o-mini'):

    if model =='gpt-4o-mini':

        # モデル利用情報の読み込み
        endpoint = os.getenv("AZURE_ENDPOINT_URL")
        subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("OPENAI_API_VERSION_4O_MINI")
        deployment = os.getenv("DEPLOYMENT_NAME_4O_MINI")  # 使用するモデルを指定

        # LLMの初期化 Runnable対応
        llm = AzureChatOpenAI(
            deployment_name=deployment,  # Azure OpenAI のモデル名
            azure_endpoint=endpoint,
            api_key=subscription_key,
            api_version=api_version
        )

    return llm

# QAの生成
def generateQA(text, model='gpt-4o-mini', encoding_name='text-embedding-3-large', chunk=512, overlap=128, num_questions=5):
    
    # エンコードタイプを設定
    encoding = set_encoding_type(encoding_name)    

    # スプリッターの初期化
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        separators=["\n", "。", "、", " "],
        encoding_name=encoding,
        chunk_size=chunk,
        chunk_overlap=overlap,
    )

    # 固定長分割
    def split_text_as_string(text):
        docs = text_splitter.create_documents([text])
        return [doc.page_content for doc in docs]

    # 再帰的分割 with ランダム
    def split_text_with_sampling(text):

        # テキストを分割
        docs = text_splitter.create_documents([text])
        split_texts = [doc.page_content for doc in docs]

        # 分割されたテキストをランダムに num_questions 個ピックアップ
        if num_questions < len(split_texts):
            selected_texts = random.sample(split_texts, num_questions)
        else:
            # 十分な数がない場合は全て
            selected_texts = split_texts  

        return selected_texts

# 固定長分割の場合はこちら
#    split_text = RunnableLambda(split_text_as_string)

    # テキスト分割をRunnnable化
    split_text = RunnableLambda(split_text_with_sampling)

    # LLMモデルの初期化 Runnable
    llm_runnable = set_llm(model='gpt-4o-mini')

    # CHAINの作成
    # promptはQA生成既定のものを仕様　langchain.chains.qa_generation.prompt import CHAT_PROMPT
    # 入力されたtxtに対し、分割を行い、分割された各txtごとにQAを作成する。
    # 分割数=QA数になる
    chain = RunnableParallel(
        text=RunnablePassthrough(),
        questions=split_text | RunnableEach(bound=prompt | llm_runnable | JsonOutputParser())
    )

    # CHAINを実行
    result = chain.invoke(text)
    
    return result.get('questions', [])

# 実行例
if __name__ == "__main__":

    import fitz  # PyMuPDF を使用
    # PDF からテキストを抽出する関数
    def extract_text_from_pdf(pdf_path):
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text() for page in doc)
        return text

    from dotenv import load_dotenv
    # .envからAOAI接続ようのパラメータを読み込み環境変数にセット
    load_dotenv()

    pdf_text = extract_text_from_pdf("C:\\Develop\\python\\GenerativeAI\\Spliter\\doc\\test.pdf")
    results = generateQA(text=pdf_text, model="gpt-4o", chunk=1024, overlap=128, num_questions=10)

    for qa in results:
        print(qa)
