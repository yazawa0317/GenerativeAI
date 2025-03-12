import os
import openai
import json
import pandas as pd
import shutil
import requests

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


@st.cache_data
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
            file_content = extract_text_from_pdf(file_path)
            all_text += file_content
        elif file_extension == ".txt":
            file_content = extract_text_detect_encode(file_path)
            all_text += file_content
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
    st.info("`Splitting doc ...`")
    if split_method == "RecursiveTextSplitter":
        split_text = recursive_split_text(text=text, encoding_name=encoding_name, chunk_size=chunk_size, overlap=overlap)
    elif split_method == "CharacterTextSplitter":
        split_text = fixlen_split_text(text=text, encoding_name=encoding_name, chunk_size=chunk_size, overlap=overlap)
    else:
        st.warning("`Split method not recognized. Using RecursiveCharacterTextSplitter`", icon="⚠️")
        split_text = recursive_split_text(text=text,  encoding_name=encoding_name, chunk_size=chunk_size, overlap=overlap)
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

        client = AzureChatOpenAI(
            azure_deployment=model_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
            api_version=api_version,
            verbose=True
        )

    return client

@st.cache_resource
def make_retriever(retriever_type, embedding_type, target_index, num_neighbors: int = 5, score_threshold: int = 0.6):
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

def make_chain(llm, retriever) -> RetrievalQA:
    """
    Make chain
    @param llm: model
    @param retriever: retriever
    @param retriever_type: retriever type
    @return: chain (or return retriever for Llama-Index)
    """
    st.info("`Making chain ...`")

#        filter_runnable = RunnableLambda(lambda x: {"question": x["question"]})

    prompt = CHAT_COMPL_PROMPT

    qa = (
        # RunnableLambda(lambda x: {"context": retriever, "question": x["question"]})
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
#            | filter_runnable
        | prompt
        | llm  # ここで LLM を直接使う
        | StrOutputParser()
        | RunnableLambda(lambda output: {'result': output})
    )
    # qa = {"context": retriever,"question": RunnablePassthrough()} | prompt | llm | StrOutputParser() | RunnableLambda(lambda output: {'result': output})

    return qa

def grade_model_answer(predicted_dataset: List, predictions: List, grade_answer_prompt: str) -> List:
    """
    Grades the distilled answer based on ground truth and model predictions.
    @param predicted_dataset: A list of dictionaries containing ground truth questions and answers.
    @param predictions: A list of dictionaries containing model predictions for the questions.
    @param grade_answer_prompt: The prompt level for the grading. Either "Fast" or "Full".
    @return: A list of scores for the distilled answers.
    """
    # Grade the distilled answer
    st.info("`Grading model answer ...`")
    # Set the grading prompt based on the grade_answer_prompt parameter
    if grade_answer_prompt == "Fast":
        prompt = GRADE_ANSWER_PROMPT_FAST
    elif grade_answer_prompt == "Descriptive w/ bias check":
        prompt = GRADE_ANSWER_PROMPT_BIAS_CHECK
    elif grade_answer_prompt == "OpenAI grading prompt":
        prompt = GRADE_ANSWER_PROMPT_OPENAI
    else:
        prompt = GRADE_ANSWER_PROMPT

    endpoint = os.getenv("AZURE_ENDPOINT_URL")
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("OPENAI_API_VERSION")

    llm_runnable = AzureChatOpenAI(
        azure_deployment='gpt-4o-mini',
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version=api_version,
        temperature=0
    )

#    prompt_runnable = ChatPromptTemplate.from_template(prompt)

    # Create an evaluation chain
#    eval_chain = prompt | llm_runnable | StrOutputParser()
    eval_chain = QAEvalChain.from_llm(
    llm=llm_runnable,  # LLM
    prompt=prompt
    )


    # Evaluate the predictions and ground truth using the evaluation chain
    graded_outputs = eval_chain.evaluate(
        predicted_dataset,
        predictions,
        question_key="question",
        prediction_key="result"
    )

    return graded_outputs


def grade_model_retrieval(gt_dataset: List, predictions: List, grade_docs_prompt: str):
    """
    Grades the relevance of retrieved documents based on ground truth and model predictions.
    @param gt_dataset: list of dictionaries containing ground truth questions and answers.
    @param predictions: list of dictionaries containing model predictions for the questions
    @param grade_docs_prompt: prompt level for the grading. Either "Fast" or "Full"
    @return: list of scores for the retrieved documents.
    """
    # Grade the docs retrieval
    st.info("`Grading relevance of retrieved docs ...`")

    # Set the grading prompt based on the grade_docs_prompt parameter
    prompt = GRADE_DOCS_PROMPT_FAST if grade_docs_prompt == "Fast" else GRADE_DOCS_PROMPT

    # Create an evaluation chain
    endpoint = os.getenv("AZURE_ENDPOINT_URL")
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("OPENAI_API_VERSION")

    llm_runnable = AzureChatOpenAI(
        azure_deployment='gpt-4o-mini',
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version=api_version,
        temperature=0
    )

#    prompt_runnable = ChatPromptTemplate.from_template(prompt)

    # Create an evaluation chain
    eval_chain = QAEvalChain.from_llm(
    llm=llm_runnable,  # LLM
    prompt=prompt
    )

    # Evaluate the predictions and ground truth using the evaluation chain
    graded_outputs = eval_chain.evaluate(
        gt_dataset,
        predictions,
        question_key="question",
        prediction_key="result"
    )
    return graded_outputs


def run_evaluation(chain, retriever, eval_set, grade_prompt, retriever_type, num_neighbors):
    """
    Runs evaluation on a model's performance on a given evaluation dataset.
    @param chain: Model chain used for answering questions
    @param retriever:  Document retriever used for retrieving relevant documents
    @param eval_set: List of dictionaries containing questions and corresponding ground truth answers
    @param grade_prompt: String prompt used for grading model's performance
    @param retriever_type: String specifying the type of retriever used
    @param num_neighbors: Number of neighbors to retrieve using the retriever
    @return: A tuple of four items:
    - answers_grade: A dictionary containing scores for the model's answers.
    - retrieval_grade: A dictionary containing scores for the model's document retrieval.
    - latencies_list: A list of latencies in seconds for each question answered.
    - predictions_list: A list of dictionaries containing the model's predicted answers and relevant documents for each question.
    """
    st.info("`Running evaluation ...`")
    predictions_list = []
    retrieved_docs = []
    gt_dataset = []
    latencies_list = []

    for data in eval_set:

        # Get answer and log latency
        start_time = time.time()
        if retriever_type != "Llama-Index":
            # invokeのdataはstrじゃないとエラーになる
            predictions_list.append(chain.invoke(data['question']))
        elif retriever_type == "Llama-Index":
            answer = chain.query(data["question"], similarity_top_k=num_neighbors, response_mode="tree_summarize",
                                 use_async=True)
            predictions_list.append({"question": data["question"], "answer": data["answer"], "result": answer.response})
        gt_dataset.append(data)
        end_time = time.time()
        elapsed_time = end_time - start_time
        latencies_list.append(elapsed_time)

        # Retrieve docs
        retrieved_doc_text = ""
        if retriever_type == "Llama-Index":
            for i, doc in enumerate(answer.source_nodes):
                retrieved_doc_text += "Doc %s: " % str(i + 1) + doc.node.text + " "

        else:
            docs = retriever.invoke(data["question"])
            for i, doc in enumerate(docs):
                retrieved_doc_text += "Doc %s: " % str(i + 1) + doc[0].page_content + " "

        retrieved = {"question": data["question"], "answer": data["answer"], "result": retrieved_doc_text}
        retrieved_docs.append(retrieved)

    # Grade
    answers_grade = grade_model_answer(gt_dataset, predictions_list, grade_prompt)
    retrieval_grade = grade_model_retrieval(gt_dataset, retrieved_docs, grade_prompt)
    return answers_grade, retrieval_grade, latencies_list, predictions_list

def cleanup(faiss_index_path):
    st.info("`Cleanup ...`")
    if os.path.exists(faiss_index_path):
        shutil.rmtree(faiss_index_path)

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

    # アプリケーション
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
            eval_set = generate_eval(text, num_eval_questions, 1000, embeddings)
        else:
            eval_set = json.loads(uploaded_eval_set.read())
        # Split text
        splits = split_texts(text, embeddings, chunk_chars, overlap, split_method)
        # Make LLM
        llm = make_llm(model)
        # Make vector DB
        retriever = make_retriever(retriever_type, embeddings, target_index)
        # Make chain
        qa_chain = make_chain(llm, retriever)
        # Grade model
        graded_answers, graded_retrieval, latency, predictions = run_evaluation(qa_chain, retriever, eval_set, grade_prompt,
                                                                        retriever_type, num_neighbors)

        e = pd.DataFrame()
        e['question'] = [g['question'] for g in eval_set]    
        e['answer'] = [g['answer'] for g in eval_set]    

        # Assemble outputs
    #    d = pd.DataFrame(predictions)
        d = pd.DataFrame()
        d['question'] = [g['question'] for g in eval_set]    
        d['answer'] = [g['result'] for g in predictions]    
        d['answer score'] = [g['results'] for g in graded_answers]
        d['docs score'] = [g['results'] for g in graded_retrieval]
        d['latency'] = latency

        # Summary statistics
        mean_latency = d['latency'].mean()
        correct_answer_count = len([text for text in d['answer score'] if "INCORRECT" not in text])
        correct_docs_count = len([text for text in d['docs score'] if "Context is relevant: True" in text])
        percentage_answer = (correct_answer_count / len(graded_answers)) * 100
        percentage_docs = (correct_docs_count / len(graded_retrieval)) * 100

        st.subheader("`QA SET`")
        st.info(
            "`I will grade the chain based on: 1/ the relevance of the retrived documents relative to the question and 2/ "
            "the summarized answer relative to the ground truth answer. You can see (and change) to prompts used for "
            "grading in text_utils`")
        st.dataframe(data=e, use_container_width=True)


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
        summary01 = pd.concat([summary01, new_row], ignore_index=True)
        st.dataframe(data=summary01, use_container_width=True)
        st.session_state.existing_df_01 = summary01

        # Dataframe for visualization
        show = summary01.reset_index().copy()
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

        st.session_state.history.append({"page": "page01", "data": {
            "text1": "`QA SET`", "df1": e,
            "text2": "`Run Results`", "df2": d,
            "text3": "`Aggregate Results`", "df3": new_row}}) 

        # **履歴が 3 件を超えたら、古いものを削除**
        # `page01` の履歴のみ 3 件までに制限
        page01_entries = [entry for entry in st.session_state.history if entry["page"] == "page01"]
        if len(page01_entries) > 3:
            # 最も古い `page01` のエントリを削除
            for i, entry in enumerate(st.session_state.history):
                if entry["page"] == "page01":
                    del st.session_state.history[i]
                    break  # 一度削除したらループを抜ける

    # 履歴を表示
    filtered_history = [
        entry for entry in reversed(st.session_state.history)  # 最新の履歴を上に表示
        if entry["page"] == "page01"
    ]


    st.subheader("📜 実行履歴")
    for idx, entry  in enumerate(filtered_history, 1):
        with st.expander(f"履歴 {idx}"):
            for key, item in entry["data"].items():
                if isinstance(item, str):
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