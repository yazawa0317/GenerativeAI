import os
import openai
import json
import pandas as pd

# def load_docs
from typing import List
import fitz 
import chardet
#

# def generate_eval
from generateQA import generateQA
#

# def split_texts
from textspliter import fixlen_split_text, recursive_split_text
#

#
#def make_llm
from openai import AzureOpenAI
#

#def make_retriever
from makeretriver import set_embeddings, create_and_save_faiss_index
#

#def make_chain
from langchain.chains import RetrievalQA
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from dotenv import load_dotenv
# .envからAOAI接続ようのパラメータを読み込み環境変数にセット
load_dotenv()

import streamlit as st

def load_docs(files: List) -> str:
    """
    Load docs from files
    @param files: list of files to load
    @return: string of all docs concatenated
    """

    st.info("`Reading doc ...`")
    all_text = ""
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1].lower()
        if file_extension == ".pdf":
            doc = fitz.open(stream=file_path.read(), filetype="pdf")
            file_content = "\n".join([page.get_text("text") for page in doc])
            all_text += file_content
        elif file_extension == ".txt":
            raw_bytes = file_path.getvalue()  # `BytesIO` の中身を取得
            # 文字コードを自動判定
            detected = chardet.detect(raw_bytes)
            encoding = detected["encoding"] if detected["encoding"] else "utf-8"
            try:
                file_content = raw_bytes.decode(encoding)  # 判定したエンコーディングでデコード
            except UnicodeDecodeError:
                file_content = raw_bytes.decode("utf-8", errors="ignore")  # デコード失敗時は UTF-8 (無視)
            all_text += file_content
        else:
            st.warning('Please provide txt or pdf.', icon="⚠️")
    return all_text

@st.cache_data
def generate_eval(text: str, num_questions: int, chunk: int):
    """
    Generate eval set
    @param text: text to generate eval set from
    @param num_questions: number of questions to generate
    @param chunk: chunk size to draw question from in the doc
    @return: eval set as JSON list
    """
    st.info("`Generating eval set ...`")
    try:
        result = generateQA(text=text, chunk=1024, num_questions=5)
        print(result)
    except:
        st.warning('Error generating question')
    
    return result

@st.cache_resource
def split_texts(text, chunk_size: int, overlap, split_method: str):
    """
    Split text into chunks
    @param text: text to split
    @param chunk_size:
    @param overlap:
    @param split_method:
    @return: list of str splits
    """
    st.info("`Splitting doc ...`")
    if split_method == "RecursiveTextSplitter":
        split_text = recursive_split_text(text=text, chunk_size=chunk_size, overlap=overlap)
    elif split_method == "CharacterTextSplitter":
        split_text = fixlen_split_text(text=text, chunk_size=chunk_size, overlap=overlap)
    else:
        st.warning("`Split method not recognized. Using RecursiveCharacterTextSplitter`", icon="⚠️")
        split_text = recursive_split_text(text=text, chunk_size=chunk_size, overlap=overlap)
    return split_text

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

        client = AzureOpenAI(
            azure_deployment=model_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
            api_version=api_version,
        )

    return client

@st.cache_resource
def make_retriever(splits, retriever_type, embedding_type, num_neighbors, _llm):
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
    if embedding_type == "OpenAI":
        embedding = set_embeddings(embedding_type='OpenAI')

    else:
        st.warning("`Embedding type not recognized. Using OpenAI`", icon="⚠️")
        embedding = set_embeddings(embedding_type='OpenAI')

    # Select retriever
    if retriever_type == "similarity-search":
        try:
            vector_store, faiss_index_path = create_and_save_faiss_index(splits, embedding)
        except ValueError:
            st.warning("`Error using OpenAI embeddings (disallowed TikToken token in the text). Using HuggingFace.`",
                       icon="⚠️")
            # vector_store = FAISS.from_texts(splits, HuggingFaceEmbeddings())
        retriever_obj = vector_store.as_retriever(k=num_neighbors)

    # elif retriever_type == "SVM":
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

def make_chain(llm, retriever, retriever_type: str) -> RetrievalQA:
    """
    Make chain
    @param llm: model
    @param retriever: retriever
    @param retriever_type: retriever type
    @return: chain (or return retriever for Llama-Index)
    """
    st.info("`Making chain ...`")
    if retriever_type == "Llama-Index":
        qa = retriever
    else:
        qa = RetrievalQA.from_chain_type(llm,
                                         chain_type="stuff",
                                         retriever=retriever,
                                         input_key="question")
    return qa

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
                      "gpt-4",
                      "anthropic"),
                     index=0)

    retriever_type = st.radio("`Choose retriever`",
                              ("TF-IDF",
                               "SVM",
                               "Llama-Index",
                               "similarity-search"),
                              index=3)

    num_neighbors = st.select_slider("`Choose # chunks to retrieve`",
                                     options=[3, 4, 5, 6, 7, 8])

    embeddings = st.radio("`Choose embeddings`",
                          ("HuggingFace",
                           "OpenAI"),
                          index=1)

    grade_prompt = st.radio("`Grading style prompt`",
                            ("Fast",
                             "Descriptive",
                             "Descriptive w/ bias check",
                             "OpenAI grading prompt"),
                            index=0)

    submitted = st.form_submit_button("Submit evaluation")

# App
st.header("`Auto-evaluator`")
st.info(
    "`I am an evaluation tool for question-answering. Given documents, I will auto-generate a question-answer eval "
    "set and evaluate using the selected chain settings. Experiments with different configurations are logged. "
    "Optionally, provide your own eval set (as a JSON, see docs/karpathy-pod-eval.json for an example).`")

with st.form(key='file_inputs'):
    uploaded_file = st.file_uploader("`Please upload a file to evaluate (.txt or .pdf):` ",
                                     type=['pdf', 'txt'],
                                     accept_multiple_files=True)

    uploaded_eval_set = st.file_uploader("`[Optional] Please upload eval set (.json):` ",
                                         type=['json'],
                                         accept_multiple_files=False)

    submitted = st.form_submit_button("Submit files")

if uploaded_file:

    # Load docs
    text = load_docs(uploaded_file)
    # Generate num_eval_questions questions, each from context of 3k chars randomly selected
    if not uploaded_eval_set:
#        eval_set = generate_eval(text, num_eval_questions, 3000)
        eval_set = generate_eval(text, 5, 3000)
    else:
        eval_set = json.loads(uploaded_eval_set.read())
    # Split text
    splits = split_texts(text, chunk_chars, overlap, split_method)
    # Make LLM
    llm = make_llm(model)
    # Make vector DB
    retriever = make_retriever(splits, retriever_type, embeddings, num_neighbors, llm)
    # Make chain
    qa_chain = make_chain(llm, retriever, retriever_type)
    # Grade model
    graded_answers, graded_retrieval, latency, predictions = run_evaluation(qa_chain, retriever, eval_set, grade_prompt,
                                                                      retriever_type, num_neighbors)

    # Assemble outputs
    d = pd.DataFrame(predictions)
    d['answer score'] = [g['text'] for g in graded_answers]
    d['docs score'] = [g['text'] for g in graded_retrieval]
    d['latency'] = latency

    # Summary statistics
    mean_latency = d['latency'].mean()
    correct_answer_count = len([text for text in d['answer score'] if "INCORRECT" not in text])
    correct_docs_count = len([text for text in d['docs score'] if "Context is relevant: True" in text])
    percentage_answer = (correct_answer_count / len(graded_answers)) * 100
    percentage_docs = (correct_docs_count / len(graded_retrieval)) * 100

    st.subheader("`Run Results`")
    st.info(
        "`I will grade the chain based on: 1/ the relevance of the retrived documents relative to the question and 2/ "
        "the summarized answer relative to the ground truth answer. You can see (and change) to prompts used for "
        "grading in text_utils`")
    st.dataframe(data=d, use_container_width=True)

    # Accumulate results
    st.subheader("`Aggregate Results`")
    st.info(
        "`Retrieval and answer scores are percentage of retrived documents deemed relevant by the LLM grader ("
        "relative to the question) and percentage of summarized answers deemed relevant (relative to ground truth "
        "answer), respectively. The size of point correponds to the latency (in seconds) of retrieval + answer "
        "summarization (larger circle = slower).`")
    new_row = pd.DataFrame({'chunk_chars': [chunk_chars],
                            'overlap': [overlap],
                            'split': [split_method],
                            'model': [model],
                            'retriever': [retriever_type],
                            'embedding': [embeddings],
                            'num_neighbors': [num_neighbors],
                            'Latency': [mean_latency],
                            'Retrieval score': [percentage_docs],
                            'Answer score': [percentage_answer]})
    summary = pd.concat([summary, new_row], ignore_index=True)
    st.dataframe(data=summary, use_container_width=True)
    st.session_state.existing_df = summary

    # Dataframe for visualization
    show = summary.reset_index().copy()
    show.columns = ['expt number', 'chunk_chars', 'overlap',
                    'split', 'model', 'retriever', 'embedding', 'num_neighbors', 'Latency', 'Retrieval score',
                    'Answer score']
    show['expt number'] = show['expt number'].apply(lambda x: "Expt #: " + str(x + 1))
    c = alt.Chart(show).mark_circle().encode(x='Retrieval score',
                                             y='Answer score',
                                             size=alt.Size('Latency'),
                                             color='expt number',
                                             tooltip=['expt number', 'Retrieval score', 'Latency', 'Answer score'])
    st.altair_chart(c, use_container_width=True, theme="streamlit")