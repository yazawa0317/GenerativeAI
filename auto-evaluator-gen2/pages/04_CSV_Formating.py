import os
import openai
import json
import pandas as pd
import shutil
import requests
import io
import uuid
import tiktoken

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

@st.cache_data
def csv_load(files: List):
    """
    Load docs from files
    @param files: list of files to load
    @return: string of all docs concatenated
    """
    st.info("`Reading doc ....`")
    all_csv = []
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1].lower()
        if file_extension == ".csv":
            df = pd.read_csv(file_path)
            if df.empty or df.columns.tolist() == [None]:
                df = pd.DataFrame(columns=["default_column"])
            all_csv.append((file_path.name,df))
        else:
            st.warning('Please provide txt or pdf.', icon="⚠️")
    return all_csv

def csv_convert(header_exist, df):
    st.info("`Splitting doc ...`")
    content_list = []
    oversize_content_list=[]
    for index, row in df.iterrows():
        markdown_content = ""
        for column in df.columns:
            if header_exist == 'True':
                markdown_content += f"## {column}\n"
                markdown_content += f"{row[column]}\n"
            else:
                markdown_content += f"{row[column]}\n"
        tokens = num_tokens_from_string(markdown_content, "gpt-3.5-turbo")
        if tokens < 8000:
            content_list.append(markdown_content.strip())  # stripで末尾の空白を削除

        else:
            oversize_content_list.append(markdown_content.strip())  # stripで末尾の空白を削除

    return content_list, oversize_content_list


def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

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

    # ロゴ
    # logo = '.streamlit\\logo.PNG'
    # st.logo(f"{logo}")

    # サイドバー
    with st.sidebar.form("user_input"):

        header_exist = st.radio("`with header`",
                                ("True",
                                "false"),
                                index=0)

        submitted = st.form_submit_button("Submit evaluation")

    # アプリケーション
    st.title("CSV整形")
    st.info(
        "`I am an evaluation tool for question-answering. Given documents, I will auto-generate a question-answer eval "
        "set and evaluate using the selected chain settings. Experiments with different configurations are logged. "
        "Optionally, provide your own eval set (as a JSON, see docs/karpathy-pod-eval.json for an example).`")

    with st.form(key='file_inputs'):
        uploaded_file = st.file_uploader("`Please upload a file to evaluate (.csv):` ",
                                        type=['csv'],
                                        accept_multiple_files=True)

        submitted = st.form_submit_button("Submit files")

    if uploaded_file:

        # Load docs
        docs = csv_load(uploaded_file)

        dfs = []

        for file_name, file_content in docs:
       # Split text
            file_content_fromated, oversize_content_list = csv_convert(header_exist, file_content)
            formated_df = pd.DataFrame({'content': file_content_fromated})
            oversize_df = pd.DataFrame({'content': oversize_content_list})
            # DataFrame に変換（カラム名を 'content' にする）
            dfs.append((file_name, "分割したデータ", formated_df))

            if oversize_content_list:
                oversize_file_name = os.path.splitext(file_name)[0] + "_oversize.csv"
                dfs.append((f"{oversize_file_name}_oversize", "サイズ超過データ", oversize_df))

        st.markdown("<h5 style='color:#808080;'>🕞 CSV出力</h5>",unsafe_allow_html=True)

        for file_name, label, df in dfs:
            # メモリ上に CSV を保存
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, encoding="utf-8")
            csv_data = csv_buffer.getvalue()

            file_name = os.path.splitext(file_name)[0].lower() + '.csv'
            unique_key = f"download_{file_name}_{uuid.uuid4()}"

            # CSV ダウンロードボタン
            st.download_button(
                label=f"📥 {label} ({file_name}) をダウンロード",
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