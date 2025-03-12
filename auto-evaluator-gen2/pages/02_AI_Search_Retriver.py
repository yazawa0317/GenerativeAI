import os
import streamlit as st
import requests
import pandas as pd

from langchain_openai import AzureChatOpenAI
#def make_chain
from langchain.chains import RetrievalQA
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
)
#def make_retriever
from tools.makeretriver import set_embeddings, similarity_search_to_aisearch


from dotenv import load_dotenv
# .envからAOAI接続ようのパラメータを読み込み環境変数にセット
load_dotenv()

def get_ai_serach_indexes():
    azure_search_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
    azure_search_key = os.getenv("AZURE_SEARCH_KEY")
    azure_search_api_version = os.getenv("AZURE_SEARCH_API_VERSION")

    headers = {
        "Content-Type": "application/json",
        "api-key": azure_search_key
    }

    try:
        r = requests.get(azure_search_endpoint + "/indexes?api-version=" + azure_search_api_version, headers=headers)
        r.raise_for_status()  # 4xx, 5xx エラーなら例外発生
        return [index["name"] for index in r.json()["value"]]

    except Exception as e:
        raise (f"インデックス情報が取得できませんでした: {e}") 

@st.cache_resource
def make_llm(model_version: str = 'gpt-4o-mini'):
    """
    Make LLM from model version
    @param model_version: model_version
    @return: LLN
    """
    if model_version == "gpt-4o-mini":

        endpoint = os.getenv("AZURE_ENDPOINT_URL")
        subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("OPENAI_API_VERSION")

        client = AzureChatOpenAI(
            azure_deployment=model_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
            api_version=api_version,
            verbose=True
        )

    return client

@st.cache_resource
def make_retriever(retriever_type, embedding_type, target_index, num_neighbors: int, score_threshold: int = 0.6):
    """
    Make document retriever
    @param splits: list of str splits
    @param retriever_type: retriever type
    @param embedding_type: embedding type
    @param num_neighbors: number of neighbors for retrieval
    @param _llm: model
    @return: retriever
    """
    st.info("`Making retriever ...`")
    # Set embeddings

    embedding = set_embeddings(embedding_type)

    # Select retriever
    if retriever_type == "similarity-search":
        try:
            vector_store = similarity_search_to_aisearch(embedding, target_index)
        except ValueError:
            st.warning("`Error using OpenAI embeddings (disallowed TikToken token in the text). Using HuggingFace.`",
                       icon="⚠️")
            # vector_store = FAISS.from_texts(splits, HuggingFaceEmbeddings())
        retriever_obj = RunnableLambda(vector_store.similarity_search_with_relevance_scores).bind(k=num_neighbors,score_threshold=score_threshold)

    ## elif retriever_type == "SVM":
    #     retriever_obj = SVMRetriever.from_texts(splits, embedding)
    # elif retriever_type == "TF-IDF":
    #     retriever_obj = TFIDFRetriever.from_texts(splits)
    # elif retriever_type == "Llama-Index":
    #     documents = [Document(t, LangchainEmbedding(embedding)) for t in splits]
    #     llm_predictor = LLMPredictor(llm)
    #     context = ServiceContext.from_defaults(chunk_size_limit=512, llm_predictor=llm_predictor)
    #     d = 1536
    #     faiss_index = faiss.IndexFlatL2(d)
    #     retriever_obj = GPTFaissIndex.from_documents(documents, faiss_index=faiss_index, service_context=context)
    # else:
    #     st.warning("`Retriever type not recognized. Using SVM`", icon="⚠️")
    #     retriever_obj = SVMRetriever.from_texts(splits, embedding)
    return retriever_obj

def retrive_chunk(retriever, prompt):
    st.info("`Retriaving ...`")

    return retriever.invoke(prompt)

def main():

    if "existing_df" not in st.session_state:
        summary02 = pd.DataFrame(columns=['index',
                                        'Retrieval score',
                                        'doc_name',
                                        'content'
                                        ])
        st.session_state.existing_df = summary02
    else:
        summary02 = st.session_state.existing_df

    # ロゴ
    # logo = '.streamlit\\logo.PNG'
    # st.logo(f"{logo}")

    # サイドバー
    with st.sidebar.form("user_input"):
        num_eval_questions = st.select_slider("`Number of eval questions`",
                                            options=[1, 5, 10, 15, 20], value=5)

        chunk_chars = st.select_slider("`Choose chunk size for splitting`",
                                    options=[500, 750, 1000, 1500, 2000], value=1000)

        overlap = st.select_slider("`Choose overlap for splitting`",
                                options=[0, 50, 100, 150, 200], value=100)

        split_method = st.radio("`Split method`",
                                ("RecursiveTextSplitter",
                                "CharacterTextSplitter"),
                                index=0)

        model = st.radio("`Choose model`",
                        ("gpt-4o-mini",
                        "gpt-4o"),
                        index=0)

        retriever_type = st.radio("`Choose retriever`",
                                ("TF-IDF",
                                "SVM",
                                "Llama-Index",
                                "similarity-search"),
                                index=3)

        num_neighbors = st.select_slider("`Choose # chunks to retrieve`",
                                        options=[3, 4, 5, 6, 7, 8])

        target_index = st.selectbox("`Choose index`",
                                options=get_ai_serach_indexes())


        embeddings = st.radio("`Choose embeddings`",
                            ("text-embedding-3-large",
                            "text-embedding-ada-002"),
                            index=0)

        grade_prompt = st.radio("`Grading style prompt`",
                                ("Fast",
                                "Descriptive",
                                "Descriptive w/ bias check",
                                "OpenAI grading prompt"),
                                index=0)

        submitted = st.form_submit_button("Submit evaluation")

    # タイトル
    st.title("プロンプト入力アプリ")

    # セッション状態の初期化
    if "history" not in st.session_state:
        st.session_state.history = []

    # テキストボックス（プロンプト入力）
    prompt = st.text_input("プロンプトを入力してください")

    # 送信ボタン
    if st.button("送信"):

        if not prompt:
            st.warning("プロンプトを入力してください。")

        else:
#            response = f"結果: {prompt}"  # 仮の処理（ここに処理ロジックを入れる）
            # Make LLM
#            llm = make_llm(model)
            # Make vector DB
            retriever = make_retriever(retriever_type, embeddings, target_index, num_neighbors)
            
            result = retrive_chunk(retriever, prompt)

            summary02['index'] = [target_index] * len(result)
            summary02['Retrieval score'] = [g.metadata['@search.score'] for g, score in result] 
            summary02['doc_name'] = [g.metadata['name'] for g, score in result] 
            summary02['content'] = [g.page_content  for g, score in result] 
            st.dataframe(data=summary02, use_container_width=True)
            st.session_state.history.append((prompt, summary02))  # 履歴に追加



    # 履歴を表示
    st.subheader("📜 メインメニュー（過去の入力履歴）")
    for idx, (old_prompt, old_response) in enumerate(reversed(st.session_state.history), 1):
        with st.expander(f"履歴 {idx}: {old_prompt}"):
            st.write(old_response)


if __name__ == "__main__":

    main()