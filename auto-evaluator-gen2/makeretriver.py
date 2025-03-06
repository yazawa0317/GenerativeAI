import os
import shutil
import uuid
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings


def set_embeddings(embedding_type):
    if embedding_type == "OpenAI":

        # Azure OpenAI の接続情報
        endpoint = os.getenv("AZURE_ENDPOINT_URL")
        subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
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

    else:
        # Azure OpenAI の接続情報
        endpoint = os.getenv("AZURE_ENDPOINT_URL")
        subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
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
def create_and_save_faiss_index(texts, embedding):

    # FAISS インデックスの保存先
    curdir = os.path.dirname(os.path.abspath(__file__))
    faiss_index_path = os.path.join(curdir, "faiss_index", f"index_{uuid.uuid4()}")


    """
    テキストリストを受け取り、FAISS インデックスを作成し、ローカルに保存。
    保存先パスを返す。
    """
    # FAISS インデックスを作成
    vector_store = FAISS.from_texts(texts, embedding)
    
    # FAISS インデックスをローカルに保存
    vector_store.save_local(faiss_index_path)
    print(f"FAISS インデックスを {faiss_index_path}/ に保存しました！")
    
    # 保存したインデックスを再読み込み
    vector_store = FAISS.load_local(faiss_index_path, embedding, allow_dangerous_deserialization=True)
    print("FAISS インデックスをローカルから読み込みました！")
    
    # ベクトルストアを返す
    return vector_store, faiss_index_path

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
    ]

    # FAISS インデックスを作成して返す
    embedding = set_embeddings(embedding_type='OpenAI')
    vector_store, faiss_index_path = create_and_save_faiss_index(texts, embedding)

    # クエリテキスト
    query = "強力"
    results = vector_store.similarity_search(query, k=3)

    # 検索結果を表示
    print("検索結果:")
    for i, res in enumerate(results):
        print(f"{i+1}. {res.page_content}")

    if os.path.exists(faiss_index_path):
        shutil.rmtree(faiss_index_path)
        print(f"FAISS インデックスの保存先 {faiss_index_path} を削除しました！")
    else:
        print(f"{faiss_index_path} は既に削除されています。")