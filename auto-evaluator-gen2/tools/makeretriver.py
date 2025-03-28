import os
import uuid

from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain_core.output_parsers import StrOutputParser

# エンベディングモデルの初期化
def set_embeddings(embedding_model):

    # Azure OpenAI の接続情報
    endpoint = os.getenv("AZURE_ENDPOINT_URL")
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

    if embedding_model == "text-embedding-3-large":

        api_version = os.getenv("OPENAI_API_VERSION_EMB3L")
        deployment = os.getenv("OPENAI_API_DEPLOYMENT_EMB3L")

    elif embedding_model == "text-embedding-ada-002":

        api_version = os.getenv("OPENAI_API_VERSION_ADA2")
        deployment = os.getenv("OPENAI_API_DEPLOYMENT_ADA2")

    else:
        api_version = os.getenv("OPENAI_API_VERSION_EMB3L")
        deployment = os.getenv("OPENAI_API_DEPLOYMENT_EMB3L")

    # OpenAIEmbeddings を Azure 用に設定
    embedding = AzureOpenAIEmbeddings(
        openai_api_key=subscription_key,
        openai_api_base=endpoint,
        openai_api_type="azure",
        openai_api_version=api_version,
        deployment=deployment,
    )

    return embedding

# ベクトル化するテキストをリストとして受け取る関数
def create_and_save_faiss_index(texts: list, embedding):

    # FAISS インデックスの保存先
    curdir = os.path.dirname(os.path.abspath(__file__))
    faiss_index_path = os.path.join(curdir, "faiss_index", f"index_{uuid.uuid4()}")

    #テキストリストを受け取り、FAISS インデックスを作成し、ローカルに保存。
    #保存先パスを返す。
    # FAISS インデックスを作成
    vector_store = FAISS.from_texts(texts, embedding)
    
    # FAISS インデックスをローカルに保存
    vector_store.save_local(faiss_index_path)
    
    # 保存したインデックスを再読み込み
    vector_store = FAISS.load_local(faiss_index_path, embedding, allow_dangerous_deserialization=True)
    
    # ベクトルストアとパスを返す。パスは後に削除するときに使う
    return vector_store, faiss_index_path

def similarity_search_to_aisearch(embeddings, index_name):

    azure_search_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
    azure_search_key = os.getenv("AZURE_SEARCH_KEY")

    vector_store = AzureSearch(
        embedding_function=embeddings.embed_query,
        azure_search_endpoint=azure_search_endpoint,
        azure_search_key=azure_search_key,
        index_name=index_name
    )

    return vector_store

# 使用例
if __name__ == "__main__":

    import shutil
    from dotenv import load_dotenv
    # .envから環境変数を読み込む
    load_dotenv()

    # # 外部から受け取るテキストリスト
    # texts = [
    #     "Azure OpenAI は強力な AI ツールです。",
    #     "FAISS は類似検索に適しています。",
    #     "Python で機械学習を行う方法。",
    #     "クラウド AI の利点について。",
    # ]

    # # FAISS インデックスを作成して返す
    # embedding = set_embeddings(embedding_model='text-embedding-3-large')
    # # FAISS の検索
    # vector_store, faiss_index_path = create_and_save_faiss_index(texts, embedding)

    # # クエリテキスト
    # query = "強力"
    # results = vector_store.similarity_search(query, k=3)

    # # 検索結果を表示
    # print("検索結果:")
    # for i, res in enumerate(results):
    #     print(f"{i+1}. {res.page_content}")

    # if os.path.exists(faiss_index_path):
    #     shutil.rmtree(faiss_index_path)
    #     print(f"FAISS インデックスの保存先 {faiss_index_path} を削除しました！")
    # else:
    #     print(f"{faiss_index_path} は既に削除されています。")

    from langchain_core.runnables import (
        RunnableLambda,
        RunnablePassthrough,
    )

    from text_utils import (
        GRADE_DOCS_PROMPT, 
        GRADE_ANSWER_PROMPT, 
        GRADE_DOCS_PROMPT_FAST, 
        GRADE_ANSWER_PROMPT_FAST, 
        GRADE_ANSWER_PROMPT_BIAS_CHECK, 
        GRADE_ANSWER_PROMPT_OPENAI, 
        CHAT_COMPL_PROMPT
        )

    from langchain_openai import AzureChatOpenAI

    endpoint = os.getenv("AZURE_ENDPOINT_URL")
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("OPENAI_API_VERSION")

    llm = AzureChatOpenAI(
        azure_deployment='gpt-4o-mini',
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version=api_version,
        verbose=True
    )

    # AI Searchの検索
    embedding = set_embeddings(embedding_model='text-embedding-3-large')
    vector_store = similarity_search_to_aisearch(embedding, index_name='yazawa-index01')
    # retriever = vector_store.as_retriever(search_type="similarity", k=1)
    retriever = RunnableLambda(vector_store.similarity_search_with_relevance_scores).bind(k=1,score_threshold=0.5)

    prompt = CHAT_COMPL_PROMPT
#    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser() | RunnableLambda(lambda output: {'result': output})
#    result = rag_chain.invoke("When did I visited Asia?")
    result2 = retriever.invoke("給与")
#    print(result2)
#    retrieved_doc_text = ""

    for doc, score in result2:
        print(doc.metadata['@search.score'])
    del retriever
    del vector_store
        # 検索を実行
    # question = "When did I visited Asia?"
    # indexes = ['yazawa-index01']
    # order_result= similarity_search_to_aisearch(embedings=embedding,
    #                                         query=question,
    #                                         indexes=indexes,
    #                                         similarity_k=3,
    #                                         k=10,
    #                                         score = 0.5)
    
    # print(order_result)

#     # 検索結果を出力
#     for id, content in order_result.items():
# #        title = str(content['title']) 
#         score = str(round(content['score'],2))
#         print( " - score: " + score )
# #        print("title: " + title + " - score: " + score )
#         print(content['content'])  