import os
import openai
import json
import pandas as pd
import shutil
import requests
import io
import uuid
import random

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
def generate_eval(text: str, num_questions: int, chunk: int, embeddings):
    """
    Generate eval set
    @param text: text to generate eval set from
    @param num_questions: number of questions to generate
    @param chunk: chunk size to draw question from in the doc
    @return: eval set as JSON list
    """
    st.info("`Generating eval set ...`")
    try:
        result = generateQA(text=text, chunk=chunk, num_questions=num_questions, encoding_name=embeddings)
    except:
        st.warning('Error generating question')
    
    return result

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


    # ロゴ
    # logo = '.streamlit\\logo.PNG'
    # st.logo(f"{logo}")

    # サイドバー
    with st.sidebar.form("user_input"):

        num_eval_questions = st.select_slider("`Number of eval questions`",
                                            options=[1, 5, 10, 15, 20], value=5)

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
        text = ''
        eval_set = []
        for file_name, content in docs:
            text += content
            result = generate_eval(text, num_eval_questions, 1000, embeddings)
            result = [{**item, "file_name": file_name} for item in result]
            eval_set += result


        selected_qa = random.sample(eval_set, num_eval_questions)

        e = pd.DataFrame()
        e['file_name'] = [g['file_name'] for g in selected_qa]  
        e['question'] = [g['question'] for g in selected_qa]    
        e['answer'] = [g['answer'] for g in selected_qa]    
        
        st.subheader("`QA SET`")
        st.info(
            "`I will grade the chain based on: 1/ the relevance of the retrived documents relative to the question and 2/ "
            "the summarized answer relative to the ground truth answer. You can see (and change) to prompts used for "
            "grading in text_utils`")
        st.dataframe(data=e, use_container_width=True)

        selected_qa = [{k: v for k, v in item.items() if k != "file_name"} for item in selected_qa]
        json_data = json.dumps(selected_qa, ensure_ascii=False, indent=4)

        st.title("JSON ダウンロード")

        st.write("以下のボタンをクリックすると JSON ファイルをダウンロードできます。")

        # ダウンロードボタン
        st.download_button(
            label="JSON をダウンロード",
            data=json_data,
            file_name="eval_set.json",
            mime="application/json"
        )

        st.session_state.history.append({"page": 'page05', "data": {
            "text1": "`QA SET`", "df1": e, 'QAFile': json_data}})

    # 履歴を表示
    filtered_history = [
        entry for entry in reversed(st.session_state.history)  # 最新の履歴を上に表示
        if entry["page"] == "page05"
    ]

    # **履歴が 3 件を超えたら、古いものを削除**
    # `page01` の履歴のみ 3 件までに制限
    page05_entries = [entry for entry in st.session_state.history if entry["page"] == "page05"]
    if len(page05_entries) > 3:
        # 最も古い `page01` のエントリを削除
        for i, entry in enumerate(st.session_state.history):
            if entry["page"] == "page05":
                del st.session_state.history[i]
                break  # 一度削除したらループを抜ける   

    with st.container(border=True):
        st.markdown("<h5 style='color:#808080;'>🕞 実行履歴</h5>",unsafe_allow_html=True)
        for idx, entry  in enumerate(filtered_history, 1):
            with st.expander(f"履歴 {idx}"):
                for key, item in entry["data"].items():
                    if isinstance(item, str):
                        if key == "QAFile":
                            with st.container():
                                st.code(item, language='json')
                        else:
                            st.subheader(item)  # 文字列を表示
                    elif isinstance(item, pd.DataFrame):
                        st.dataframe(item)  # DataFrame を表示

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