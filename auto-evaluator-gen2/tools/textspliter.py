import os
import pandas as pd
import tiktoken

from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter


def set_encoding_type(model):
    # エンコードタイプを設定
    if model in ['gpt-4o', 'gpt-4o-mini']:
        encoding_name= 'o200k_base'
    elif model in ['text-embedding-ada-002', 'text-embedding-3-small', 'text-embedding-3-large']:
        encoding_name= 'cl100k_base'
    
    return encoding_name

# 固定長分割
def fixlen_split_text(text, encoding_name='text-embedding-3-large', chunk_size=512, overlap=0):

    # エンコードタイプを設定
    encoding = set_encoding_type(encoding_name)

    # スプリッターの初期化
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=encoding,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )

    return splitter.split_text(text)

# 再帰的分割
def recursive_split_text(text, encoding_name='text-embedding-3-large', chunk_size=512, overlap=128):

    # エンコードタイプを設定
    encoding = set_encoding_type(encoding_name)

    # スプリッターの初期化
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        separators=["\n", "。", "、", " "],
        encoding_name=encoding,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    return splitter.split_text(text)

# トークン数の検査
def count_tokens(text, encoding_name='text-embedding-3-large'):

    # エンコードタイプを設定
    encoding = set_encoding_type(encoding_name)
    tkt = tiktoken.get_encoding(encoding)

    return len(tkt.encode(text))

# 分割した結果をCSV出力
def save_chunks_to_csv(chunks, output_csv, encoding_name='text-embedding-3-large'):

    data = [{"chunk_id": i+1, "tokens": count_tokens(chunk, encoding_name), "chunk": chunk.replace("\n", " ")} for i, chunk in enumerate(chunks)]
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    

if __name__ == '__main__':

    import fitz  # PyMuPDF
    def extract_text_from_pdf(pdf_path):
        """PDFからテキストを抽出する"""
        doc = fitz.open(pdf_path)  # PDFを開く
        text = "\n".join([page.get_text("text") for page in doc])  # 全ページのテキスト取得
        return text

    # 対象ファイルのパスを指定
    docdir = os.path.dirname(os.path.abspath(__file__)) + r'\tmp\doc'
    pdf_path = os.path.join(docdir, 'sample.pdf')
    out_path = os.path.join(docdir, 'output.csv')

    # pdf
    extracted_text = extract_text_from_pdf(pdf_path)  # PDFからテキストを抽出
    chunks = fixlen_split_text(extracted_text)  # テキストを512文字で分割
#    chunks = recursive_split_text(extracted_text)  # テキストを512文字で分割

    save_chunks_to_csv(chunks, out_path)  # CSVに保存

    # 分割結果を確認
    # for i, chunk in enumerate(chunks):
    #     print(f"Chunk {i+1}:\n{chunk}\n{'-'*50}")