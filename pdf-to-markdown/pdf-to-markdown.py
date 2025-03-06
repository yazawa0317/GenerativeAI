import os 
import pymupdf4llm
import pathlib

from langchain.text_splitter import MarkdownTextSplitter, MarkdownHeaderTextSplitter

docdir = os.path.dirname(os.path.abspath(__file__)) + r'\doc'

src_doc = os.path.join(docdir, 'test2.pdf')
dst_doc = os.path.join(docdir, 'output.md')

# PDFからMarkdownを抽出
md_text = pymupdf4llm.to_markdown(src_doc)
print(md_text)
#pathlib.Path(dst_doc).write_bytes(md_text.encode())

#----

markdown_text = md_text

# headers_to_split_on = [
#     ("#", "Header 1"),
#     ("##", "Header 2"),
#     ("###", "Header 3"),
# ]

# MarkdownTextSplitterのインスタンスを生成
# ここではチャンクサイズを100に設定し、重複は0に設定しています。
markdown_splitter = MarkdownTextSplitter(chunk_size=512, chunk_overlap=128)
docs = markdown_splitter.create_documents([markdown_text])
# markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
# docs = markdown_splitter.split_text(markdown_text)
# create_documentsメソッドを使用して、マークダウンテキストを文書に分割

# for _ in docs:
#     print(_)
