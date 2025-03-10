import os
import openai
import json
import requests
from collections import OrderedDict

from langchain.embeddings import AzureOpenAIEmbeddings

from dotenv import load_dotenv
# .envからAOAI接続ようのパラメータを読み込み環境変数にセット
load_dotenv(override=True)

azure_search_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
azure_search_key = os.getenv("AZURE_SEARCH_KEY")

openai.api_type = os.getenv("OPENAI_TYPE")
openai.base_url = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")

def get_search_vector_results(query: str, indexes: list, 
                       k: int = 10, # AI Searchのインデクスから結果を取得するドキュメント数
                       similarity_k: int = 3, # ベクトル検索した結果を出力する最大数
                       ):

    azure_search_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
    azure_search_key = os.getenv("AZURE_SEARCH_KEY")
    azure_search_api_version = os.getenv("AZURE_SEARCH_API_VERSION")
    api_version = os.getenv("OPENAI_API_VERSION_EMB3L")
    deployment = os.getenv("OPENAI_API_DEPLOYMENT_EMB3L")

    # AI Searchへ接続するためのヘッダーを定義する。
    headers = {'Content-Type': 'application/json','api-key': azure_search_key}
    params = {'api-version': azure_search_api_version}

    # 検索結果を保存する辞書を作成
    agg_search_results = dict()

    # queryをvectorにする。
    embedings = AzureOpenAIEmbeddings(
        # モデルデプロイ名をいれる
        azure_deployment= "text-embedding-ada-002",
        azure_endpoint=azure_search_endpoint,
        openai_api_version=api_version ,
        openai_api_key=openai.api_key or ""
    )

    for index in indexes:

        # queryをvectorにする。
        query_vector = embedings.embed_query(query)

        # AI Search検索用のpayloadを作成する
        search_payload = {
                "select": "chunk_id, id, content, content_vector, title, name, location", # 取得するフィールドを指定
                "count": True, # 取得したドキュメントの数を取得する
                "vectors": [{"value": query_vector, "fields": "content_vector","k": k}] # ベクトル化した質問、インデックスのベクトルフィールド指定、何個のドキュメントを取得するか
            }

        # 検索を実行する。uriに検索対象のindex名を指定するようになっている。
        r = requests.post(os.environ['AZURE_SEARCH_ENDPOINT'] + "/indexes/" + index + "/docs/search",
                            data=json.dumps(search_payload), headers=headers, params=params)

        # 検索した結果を辞書に保存
        search_results = r.json()
        agg_search_results[index] = search_results
        # print("Index:", index, "Results Found: {}, Results Returned: {}".format(search_results['@odata.count'], len(search_results['value'])))
        # Index: yazawa-dev-index01-vector Results Found: 5, Results Returned: 5

    # 検索結果を保存する辞書を作成
    content = dict()
    ordered_content = OrderedDict()

    # 検索結果からスコアの良いものを最終的なコンテンツに保存
    for index,search_results in agg_search_results.items():
        for result in search_results['value']:
            if result['@search.score'] > 0.7: # スコアの閾値　質のいいものだけをピックアップ
                content[result['chunk_id']]={
                                        "title": result['title'],
                                        "name": result['name'],
                                        "location": result['location'],
                                        "content": result['content'], 
                                        "score": result['@search.score'],
                                        "index": index
                                        }

    # search scoreについて
    # ベクトル検索は、ベクトル化した質問をAI Searchに渡すと、検索をして、類似性の高いドキュメント（スコアの高いもの）を返してくれるようになっている。
    # スコアは0~1の間で、1に近いほどスコアが高い。
    # 本来は以下のような計算を行う
    # ---------------------------------------------------------------------------------------
    # query_vector = embedings.embed_query("飛行車の最高速度は？")
    # document1_vector = embedings.embed_query("飛行者の最高速度は150キロメートルです")     
    # doc1との類似性を計算
    # cos_sim1 = dot(query_vector, document1_vector) / (norm(query_vector) * norm(document1_vector))
    # print(f"ドキュメント1と質問の類似度:{cos_sim1}")
    # ベクトル化された質問：[-0.005822537504652993, -0.005708369929119317, -0.008494054953718445, -0.012434461234702892, -0.0185277388701657]
    # ドキュメント1と質問の類似度:0.9201409597867932　←　これがスコア★
    #    print(content)
        count = 0  # カウンター
        for id in sorted(content, key=lambda x: content[x]["score"], reverse=True):
            ordered_content[id] = content[id]
            count += 1
            if count >= similarity_k:  # コンテンツに含む結果の数
                    break

        return ordered_content

if __name__ == '__main__':

    # 検索するIndexを指定する。複数イケるようにしている。
    indexes = ["matsuyama-index01"]
    question = "出張をした場合、一日あたりいくらい支給されますか？"

    # 検索を実行
    order_result= get_search_vector_results(query=question,
                                            indexes=indexes,
                                            similarity_k=3,
                                            k=10)
    
    print(len(order_result))

    # 検索結果を出力
    for id, content in order_result.items():
#        title = str(content['title']) 
        score = str(round(content['score'],2))
        print( " - score: " + score )
#        print("title: " + title + " - score: " + score )
        print(content['content'])  