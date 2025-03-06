from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_openai import AzureChatOpenAI
import os
from makeretriver import set_embeddings, create_and_save_faiss_index
from dotenv import load_dotenv
# .envから環境変数を読み込む
load_dotenv()

# 外部から受け取るテキストリスト
texts = [
    "Azure OpenAI は強力な AI ツールです。",
    "FAISS は類似検索に適しています。",
    "Python で機械学習を行う方法。",
    "クラウド AI の利点について。",
    "ベルデータはシステムインテグレータの会社です。"
]

# FAISS インデックスを作成して返す
embedding = set_embeddings(embedding_type='OpenAI')
vector_store, faiss_index_path = create_and_save_faiss_index(texts, embedding)
retriever = vector_store.as_retriever(k=5)

model_version = "gpt-4o-mini"  # 使用するモデルを指定
endpoint = os.getenv("AZURE_ENDPOINT_URL")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("OPENAI_API_VERSION")

# 直接 AzureChatOpenAI を利用
llm = AzureChatOpenAI(
    deployment_name="gpt-4o-mini",  # Azure OpenAI のモデル名
    azure_endpoint=os.getenv("AZURE_ENDPOINT_URL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION"),
)
# プロンプト定義
prompt = ChatPromptTemplate.from_template(
    "あなたは親切なアシスタントです。日本語で回答してください。\n\n{question}"
)

# questionだけを渡すためにフィルターするランナブル
#filter_runnable = RunnableLambda(lambda x: {"question": x["question"]})
filter_runnable = RunnableLambda(lambda x: x["question"])
# チェーン構築
qa = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
#        "question": filter_runnable
    }
#    | filter_runnable  # `question`だけ渡す
    | prompt
    | llm
    | StrOutputParser()
)

# 実行例
data = {"question": "給与規程が改定されたのは何年何月何日ですか？", "answer": "2021年10月27日"}
print(type(data["question"]))
response = qa.invoke(data["question"])
print(response)