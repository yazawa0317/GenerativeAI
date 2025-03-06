from langchain.evaluation import QAEvalChain
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
import os
from text_utils import GRADE_DOCS_PROMPT, GRADE_ANSWER_PROMPT, GRADE_DOCS_PROMPT_FAST, GRADE_ANSWER_PROMPT_FAST, GRADE_ANSWER_PROMPT_BIAS_CHECK, GRADE_ANSWER_PROMPT_OPENAI

from dotenv import load_dotenv
# .envから環境変数を読み込む
load_dotenv()

# LLM と評価チェーンの作成
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

prompt = GRADE_ANSWER_PROMPT_FAST

eval_chain = QAEvalChain.from_llm(
    llm=llm,
    prompt=prompt,
    verbose=True
)

# 予測結果と正しい答えのデータ
predicted_dataset = [
    {"question": "給与規程が改定されたのは何年何月何日ですか？", "answer": "2021年10月27日"}
]

predictions = [
    {"result": "2021年10月27日"}
]

# evaluate を使って評価
graded_outputs = eval_chain.evaluate(
    predicted_dataset,  # 予測されたデータセット
    predictions,        # 実際の予測結果
    question_key="question",  # 質問に関するキー
    prediction_key="result"   # 予測結果に関するキー
)

# 評価結果を出力
print(graded_outputs)