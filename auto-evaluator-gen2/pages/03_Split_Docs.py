import os
import openai
import json
import pandas as pd
import shutil
import requests
import io
import uuid

# def load_docs
from typing import List
from tools.fileconverter import extract_text_detect_encode, extract_text_from_pdf
#

# def generate_eval
from tools.generateQA import generateQA
#

# def split_texts
from tools.textspliter import fixlen_split_text, recursive_split_text
#

#
#def make_llm
from langchain_openai import AzureChatOpenAI
#

#def make_retriever
from tools.makeretriver import set_embeddings, similarity_search_to_aisearch
#

#def make_chain
from langchain.chains import RetrievalQA
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser
#

# run evalution
import time
from tools.text_utils import (
    GRADE_DOCS_PROMPT, 
    GRADE_ANSWER_PROMPT, 
    GRADE_DOCS_PROMPT_FAST, 
    GRADE_ANSWER_PROMPT_FAST, 
    GRADE_ANSWER_PROMPT_BIAS_CHECK, 
    GRADE_ANSWER_PROMPT_OPENAI, 
    CHAT_COMPL_PROMPT
    )
from langchain.evaluation.qa import QAEvalChain
#

import altair as alt


import stconfig
import streamlit as st


st.set_page_config(**stconfig.SET_PAGE_CONFIG)
st.markdown(stconfig.HIDE_ST_STYLE, unsafe_allow_html=True)
# selected = option_menu(**const.OPTION_MENU_CONFIG)


from dotenv import load_dotenv
# .envからAOAI接続ようのパラメータを読み込み環境変数にセット
load_dotenv()

import streamlit as st

# Keep dataframe in memory to accumulate experimental results
if "existing_df_01" not in st.session_state:
    summary01 = pd.DataFrame(columns=['chunk_chars',
                                    'overlap',
                                    'split',
                                    'model',
                                    'retriever',
                                    'embedding',
                                    'num_neighbors',
                                    'Latency',
                                    'Retrieval score',
                                    'Answer score'])
    st.session_state.existing_df_01 = summary01
else:
    summary01 = st.session_state.existing_df_01

@st.cache_data
def load_docs(files: List) -> str:
    """
    Load docs from files
    @param files: list of files to load
    @return: string of all docs concatenated
    """

    all_text = []
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1].lower()
        if file_extension == ".pdf":
            file_content = extract_text_from_pdf(file_path)
            all_text.append((file_path.name,file_content))
        elif file_extension == ".txt":
            file_content = extract_text_detect_encode(file_path)
            all_text.append((file_path.name,file_content))
        else:
            st.warning('Please provide txt or pdf.', icon="⚠️")
    return all_text

@st.cache_resource
def split_texts(text, encoding_name, chunk_size: int, overlap, split_method: str):
    """
    Split text into chunks
    @param text: text to split
    @param chunk_size:
    @param overlap:
    @param split_method:
    @return: list of str splits
    """
    if split_method == "RecursiveTextSplitter":
        split_text = recursive_split_text(text=text, encoding_name=encoding_name, chunk_size=chunk_size, overlap=overlap)
    elif split_method == "CharacterTextSplitter":
        split_text = fixlen_split_text(text=text, encoding_name=encoding_name, chunk_size=chunk_size, overlap=overlap)
    else:
        st.warning("`Split method not recognized. Using RecursiveCharacterTextSplitter`", icon="⚠️")
        split_text = recursive_split_text(text=text,  encoding_name=encoding_name, chunk_size=chunk_size, overlap=overlap)
    return split_text

def main():

    # セッション状態の初期化
    if "history" not in st.session_state:
        st.session_state.history = []

    # Keep dataframe in memory to accumulate experimental results
    if "existing_df_01" not in st.session_state:
        summary01 = pd.DataFrame(columns=['chunk_chars',
                                        'overlap',
                                        'split',
                                        'model',
                                        'retriever',
                                        'embedding',
                                        'num_neighbors',
                                        'Latency',
                                        'Retrieval score',
                                        'Answer score'])
        st.session_state.existing_df_01 = summary01
    else:
        summary01 = st.session_state.existing_df_01

    # ロゴ
    # logo = '.streamlit\\logo.PNG'
    # st.logo(f"{logo}")

    # サイドバー
    with st.sidebar.form("user_input"):

        chunk_chars = st.select_slider("`Choose chunk size for splitting`",
                                    options=[500, 750, 1000, 1500, 2000], value=1000)

        overlap = st.select_slider("`Choose overlap for splitting`",
                                options=[0, 50, 100, 150, 200], value=100)

        split_method = st.radio("`Split method`",
                                ("RecursiveTextSplitter",
                                "CharacterTextSplitter"),
                                index=0)

        embeddings = st.radio("`Choose embeddings`",
                            ("text-embedding-3-large",
                            "text-embedding-ada-002"),
                            index=0)

        submitted = st.form_submit_button("Submit evaluation")

    # アプリケーション
    st.title("ドキュメント分割")
    st.info(
        "`I am an evaluation tool for question-answering. Given documents, I will auto-generate a question-answer eval "
        "set and evaluate using the selected chain settings. Experiments with different configurations are logged. "
        "Optionally, provide your own eval set (as a JSON, see docs/karpathy-pod-eval.json for an example).`")

    with st.form(key='file_inputs'):
        uploaded_file = st.file_uploader("`Please upload a file to evaluate (.txt or .pdf):` ",
                                        type=['pdf', 'txt'],
                                        accept_multiple_files=True)

        submitted = st.form_submit_button("Submit files")

    if uploaded_file:

        # Load docs
        st.info("`Reading doc ....`")
        docs = load_docs(uploaded_file)
        st.info("`Splitting doc ...`")
        dfs = []
        for file_name, file_content in docs:
       # Split text
            splits = split_texts(file_content, embeddings, chunk_chars, overlap, split_method)
            columns = ['content']
            # DataFrame に変換（カラム名を 'content' にする）
            cleaned_data = [item.replace("\n", " ").replace("\r", " ") for item in splits]
            df = pd.DataFrame(cleaned_data, columns=columns)
            dfs.append((file_name, df))
        
        st.markdown("<h5 style='color:#808080;'>🕞 CSV出力</h5>",unsafe_allow_html=True)

        for file_name, df in dfs:
            # メモリ上に CSV を保存
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, encoding="utf-8")
            csv_data = csv_buffer.getvalue()

            file_name = os.path.splitext(file_name)[0].lower() + '.csv'
            unique_key = f"download_{file_name}_{uuid.uuid4()}"

            # CSV ダウンロードボタン
            st.download_button(
                label=f"📥 {file_name} をダウンロード",
                data=csv_data,
                file_name=file_name,
                mime="text/csv",
                key=unique_key  # UUID をキーに設定し、重複を防ぐ
    )


if __name__ == "__main__":

    main()
    # import streamlit_authenticator as stauth
    # import yaml
    # docdir = os.path.dirname(os.path.abspath(__file__)) + r'\.streamlit'
    # yaml_path = os.path.join(docdir, '.config.yaml')

    # with open(yaml_path) as file:
    #     config = yaml.load(file, Loader=yaml.SafeLoader)

    # authenticator = stauth.Authenticate(
    #     config['credentials'],
    #     config['cookie']['name'],
    #     config['cookie']['key'],
    #     config['cookie']['expiry_days'],
    # )

    # authenticator.login(clear_on_submit=True)

    # # st.session_stateに変える
    # if st.session_state['authentication_status']:
    #     authenticator.logout('Logout', 'sidebar')
    #     st.write('Welcome *%s*' % (st.session_state['name']))
    #     main()
    # elif st.session_state['authentication_status'] == False:
    #     st.error('Username/password is incorrect')
    # elif st.session_state['authentication_status'] == None:
    #     st.warning('Please enter your username and password')