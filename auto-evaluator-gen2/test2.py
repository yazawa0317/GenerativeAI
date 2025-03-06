from langchain.chains import RetrievalQA  # 適切なモジュール名に置き換えてください
from openai import AzureOpenAI
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from makeretriver import set_embeddings, create_and_save_faiss_index
import os

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
            {"role": "system", "content": "あなたは親切なアシスタントです。日本語で回答してください。"},
            {"role": "user", "content": input_text}
        ]
    )
    return response.choices[0].message.content

def make_chain(model, retriever, retriever_type: str) -> RetrievalQA:
    """
    Make chain
    @param llm: model
    @param retriever: retriever
    @param retriever_type: retriever type
    @return: chain (or return retriever for Llama-Index)
    """

    if retriever_type == "Llama-Index":
        qa = retriever
    else:
        # llmをRunnableに適合させる
        llm_runnable = RunnableLambda(lambda input_text, **kwargs:  call_openai(input_text, model))
        prompt = hub.pull("rlm/rag-prompt")
        # qa = RetrievalQA.from_chain_type(
        #     llm_runnable,  # Runnableインスタンスを渡す
        #     chain_type="stuff",
        #     retriever=retriever,
        #     input_key="question"
        # )

        # qa = RetrievalQA.from_llm(llm_runnable, retriever=retriever, prompt=prompt)

        qa = (
            {
                "context": retriever,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm_runnable
            | StrOutputParser()
        )
    return qa

# 実行例
if __name__ == "__main__":


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


    model_version = "gpt-4o-mini"  # 使用するモデルを指定
    endpoint = os.getenv("AZURE_ENDPOINT_URL")
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("OPENAI_API_VERSION")

    retriever = vector_store.as_retriever(k=5)
    qa_instance = make_chain(model='gpt-4o', retriever=retriever, retriever_type="stuff")  # 適切なリトリーバータイプを指定

    # 質問を実行
    question = "ベルデータは何の会社ですか？"
#    result = qa_instance({"question": question})  # 質問を辞書形式で渡す
    result = qa_instance.invoke(question)  # 質問を辞書形式で渡す
    print(result)