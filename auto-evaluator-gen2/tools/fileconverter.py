import fitz 
import chardet

# pdfからtxtへの変換
def extract_text_from_pdf(file_path):
    doc = fitz.open(stream=file_path.read(), filetype="pdf")
    file_content = "\n".join([page.get_text("text") for page in doc])

    return file_content

def extract_text_detect_encode(file_path):
    raw_bytes = file_path.read()   # `BytesIO` の中身を取得

    # 文字コードを自動判定
    detected = chardet.detect(raw_bytes)
    encoding = detected["encoding"] if detected["encoding"] else "utf-8"
    try:
        file_content = raw_bytes.decode(encoding)  # 判定したエンコーディングでデコード
    except UnicodeDecodeError:
        file_content = raw_bytes.decode("utf-8", errors="ignore")  # デコード失敗時は UTF-8 (無視)

    return file_content

# 実行例
if __name__ == "__main__":

    pdf_text = "C:\\Develop\\python\\GenerativeAI\\Spliter\\doc\\test.pdf"
    with open(pdf_text, "rb") as f:
        content = extract_text_from_pdf(f)

    print(content)

    text = "C:\\Develop\\python\\GenerativeAI\\Spliter\\doc\\sample.txt"
    with open(text, "rb") as f:
        content = extract_text_detect_encode(f)

    print(content)