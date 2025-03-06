import os
import pandas as pd
import fitz  # PyMuPDF
import tiktoken

from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_path):
    """PDFからテキストを抽出する"""
    doc = fitz.open(pdf_path)  # PDFを開く
    text = "\n".join([page.get_text("text") for page in doc])  # 全ページのテキスト取得
    return text

def fixlen_split_text(text, chunk_size=512, overlap=128):
    """テキストを指定サイズで分割"""
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        encoding_name='o200k_base',
        chunk_size=chunk_size,
        chunk_overlap=overlap,
#        length_function=len
    )
    return splitter.split_text(text)

def recursive_split_text(text, chunk_size=512, overlap=128):
    """テキストを指定サイズで分割"""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        separators=["\n", "。", "、", " "],
        encoding_name='o200k_base',
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    return splitter.split_text(text)

def count_tokens(text, encoding_name="o200k_base"):
    """テキストのトークン数を計算"""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

def save_chunks_to_csv(chunks, output_csv):
    """分割したチャンクをCSVに保存"""
    data = [{"chunk_id": i+1, "tokens": count_tokens(chunk), "chunk": chunk.replace("\n", " ")} for i, chunk in enumerate(chunks)]
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False, encoding="utf-8")

if __name__ == '__main__':

    docdir = os.path.dirname(os.path.abspath(__file__)) + r'\doc'
    pdf_path = os.path.join(docdir, 'sample.pdf')
    out_path = os.path.join(docdir, 'output.csv')

    extracted_text = extract_text_from_pdf(pdf_path)  # PDFからテキストを抽出
    chunks = fixlen_split_text(extracted_text)  # テキストを512文字で分割
#    chunks = recursive_split_text(extracted_text)  # テキストを512文字で分割

    save_chunks_to_csv(chunks, out_path)  # CSVに保存

    # 分割結果を確認
    # for i, chunk in enumerate(chunks):
    #     print(f"Chunk {i+1}:\n{chunk}\n{'-'*50}")