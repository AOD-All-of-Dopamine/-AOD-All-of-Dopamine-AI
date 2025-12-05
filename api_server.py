# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
from flask_cors import CORS
from flasgger import Swagger
import json
from make_bigraph import (
    set_api_data, build_nodes, build_raw_genres, 
    build_bipartite, USE_API_MODE
)
from make_itemgraph_api import build_item_graph
from make_userembedding_api import build_user_embeddings
import pandas as pd

app = Flask(__name__)
CORS(app)
swagger = Swagger(app)

@app.route('/health', methods=['GET'])
def health():
    """
    헬스 체크
    ---
    tags:
      - Health
    responses:
      200:
        description: 서버가 정상 작동 중
        schema:
          properties:
            status:
              type: string
              example: "ok"
            message:
              type: string
              example: "API is running"
    """
    return jsonify({"status": "ok", "message": "API is running"})

@app.route('/build-graph', methods=['POST'])
def build_graph():
    """
    그래프 생성 API (CSV로 저장)
    ---
    tags:
      - Graph Building
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            contents:
              type: array
              description: contents.csv 데이터
              items:
                type: object
            av:
              type: array
              description: av_contents.csv 데이터 (선택사항)
              items:
                type: object
            game:
              type: array
              description: game_contents.csv 데이터 (선택사항)
              items:
                type: object
            webnovel:
              type: array
              description: webnovel_contents.csv 데이터 (선택사항)
              items:
                type: object
            raw_item:
              type: array
              description: raw_item.csv 데이터 (선택사항)
              items:
                type: object
          required:
            - contents
    responses:
      200:
        description: 그래프 생성 성공
        schema:
          properties:
            success:
              type: boolean
              example: true
            message:
              type: string
              example: "그래프가 성공적으로 생성되었습니다."
            stats:
              type: object
              properties:
                content_nodes_count:
                  type: integer
                meta_nodes_count:
                  type: integer
                edges_count:
                  type: integer
      400:
        description: 잘못된 요청
      500:
        description: 서버 오류
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "error": "JSON 데이터가 필요합니다"}), 400
        
        # API 데이터 설정
        set_api_data(
            contents_json=data.get('contents'),
            av_json=data.get('av'),
            game_json=data.get('game'),
            webnovel_json=data.get('webnovel'),
            raw_item_json=data.get('raw_item')
        )
        
        # 그래프 생성
        print("📊 그래프 생성 시작...")
        nodes = build_nodes()
        print(f"✅ 노드 생성 완료: {len(nodes)} rows")
        
        rawg = build_raw_genres(nodes)
        print(f"✅ 장르 데이터 생성 완료: {len(rawg)} rows")
        
        meta_nodes, edges = build_bipartite(nodes, rawg)
        print(f"✅ 이분 그래프 생성 완료: {len(meta_nodes)} meta nodes, {len(edges)} edges")
        
        return jsonify({
            "success": True,
            "message": "그래프가 성공적으로 생성되었습니다.",
            "stats": {
                "content_nodes_count": len(nodes),
                "meta_nodes_count": len(meta_nodes),
                "edges_count": len(edges)
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/build-graph-get-result', methods=['POST'])
def build_graph_get_result():
    """
    그래프 생성 후 결과 반환 API (메모리에서)
    ---
    tags:
      - Graph Building
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            contents:
              type: array
              description: contents.csv 데이터
              items:
                type: object
            av:
              type: array
              description: av_contents.csv 데이터 (선택사항)
              items:
                type: object
            game:
              type: array
              description: game_contents.csv 데이터 (선택사항)
              items:
                type: object
            webnovel:
              type: array
              description: webnovel_contents.csv 데이터 (선택사항)
              items:
                type: object
            raw_item:
              type: array
              description: raw_item.csv 데이터 (선택사항)
              items:
                type: object
          required:
            - contents
    responses:
      200:
        description: 그래프 생성 및 결과 조회 성공
        schema:
          properties:
            success:
              type: boolean
              example: true
            stats:
              type: object
              properties:
                content_nodes_count:
                  type: integer
                raw_genres_count:
                  type: integer
                meta_nodes_count:
                  type: integer
                edges_count:
                  type: integer
            nodes:
              type: array
              description: 콘텐츠 노드 데이터 (graph_nodes.csv)
              items:
                type: object
            raw_genres:
              type: array
              description: 장르 데이터 (content_raw_genres.csv)
              items:
                type: object
            meta_nodes:
              type: array
              description: 메타노드 (meta_nodes.csv)
              items:
                type: object
            edges:
              type: array
              description: 이분 그래프 엣지 데이터 (graph_edges_bipartite.csv)
              items:
                type: object
      400:
        description: 잘못된 요청
      500:
        description: 서버 오류
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "error": "JSON 데이터가 필요합니다"}), 400
        
        # API 데이터 설정
        set_api_data(
            contents_json=data.get('contents'),
            av_json=data.get('av'),
            game_json=data.get('game'),
            webnovel_json=data.get('webnovel'),
            raw_item_json=data.get('raw_item')
        )
        
        # 그래프 생성
        nodes = build_nodes()
        rawg = build_raw_genres(nodes)
        meta_nodes, edges = build_bipartite(nodes, rawg)
        
        # JSON으로 변환
        nodes_json = nodes.to_dict('records')
        rawg_json = rawg.to_dict('records')
        meta_nodes_json = meta_nodes.to_dict('records')
        edges_json = edges.to_dict('records')
        
        return jsonify({
            "success": True,
            "stats": {
                "content_nodes_count": len(nodes),
                "raw_genres_count": len(rawg),
                "meta_nodes_count": len(meta_nodes),
                "edges_count": len(edges)
            },
            "nodes": nodes_json,
            "raw_genres": rawg_json,
            "meta_nodes": meta_nodes_json,
            "edges": edges_json
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/build-item-graph', methods=['POST'])
def build_item_graph_endpoint():
    """
    아이템-아이템 그래프 생성 API (CSV로 저장)
    ---
    tags:
      - Graph Building
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            nodes:
              type: array
              description: graph_nodes.csv 데이터
              items:
                type: object
            raw_genres:
              type: array
              description: content_raw_genres.csv 데이터 (선택사항)
              items:
                type: object
          required:
            - nodes
    responses:
      200:
        description: 아이템 그래프 생성 성공
        schema:
          properties:
            success:
              type: boolean
              example: true
            message:
              type: string
              example: "아이템 그래프가 성공적으로 생성되었습니다."
            stats:
              type: object
              properties:
                edges_count:
                  type: integer
      400:
        description: 잘못된 요청
      500:
        description: 서버 오류
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "error": "JSON 데이터가 필요합니다"}), 400
        
        nodes_data = data.get('nodes')
        raw_genres_data = data.get('raw_genres')
        
        if not nodes_data:
            return jsonify({"success": False, "error": "nodes 데이터가 필요합니다"}), 400
        
        # DataFrame으로 변환
        nodes_df = pd.DataFrame(nodes_data)
        raw_genres_df = pd.DataFrame(raw_genres_data) if raw_genres_data else None
        
        # 아이템 그래프 생성
        print("📊 아이템 그래프 생성 시작...")
        edges = build_item_graph(nodes_df, raw_genres_df)
        
        # CSV로 저장
        import os
        BASE = r"csv 데이터\clean"
        OUT_EDGES = os.path.join(BASE, "graph_edges_item_item.csv")
        edges.to_csv(OUT_EDGES, index=False, encoding="utf-8-sig")
        print(f"✅ 아이템 그래프 저장: {OUT_EDGES}")
        
        return jsonify({
            "success": True,
            "message": "아이템 그래프가 성공적으로 생성되었습니다.",
            "stats": {
                "edges_count": len(edges)
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/build-item-graph-get-result', methods=['POST'])
def build_item_graph_get_result():
    """
    아이템-아이템 그래프 생성 후 결과 반환 API (메모리에서)
    ---
    tags:
      - Graph Building
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            nodes:
              type: array
              description: graph_nodes.csv 데이터
              items:
                type: object
            raw_genres:
              type: array
              description: content_raw_genres.csv 데이터 (선택사항)
              items:
                type: object
          required:
            - nodes
    responses:
      200:
        description: 아이템 그래프 생성 및 결과 조회 성공
        schema:
          properties:
            success:
              type: boolean
              example: true
            stats:
              type: object
              properties:
                edges_count:
                  type: integer
            edges:
              type: array
              description: 아이템-아이템 엣지 데이터 (graph_edges_item_item.csv)
              items:
                type: object
      400:
        description: 잘못된 요청
      500:
        description: 서버 오류
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "error": "JSON 데이터가 필요합니다"}), 400
        
        nodes_data = data.get('nodes')
        raw_genres_data = data.get('raw_genres')
        
        if not nodes_data:
            return jsonify({"success": False, "error": "nodes 데이터가 필요합니다"}), 400
        
        # DataFrame으로 변환
        nodes_df = pd.DataFrame(nodes_data)
        raw_genres_df = pd.DataFrame(raw_genres_data) if raw_genres_data else None
        
        # 아이템 그래프 생성
        edges = build_item_graph(nodes_df, raw_genres_df)
        
        # JSON으로 변환
        edges_json = edges.to_dict('records')
        
        return jsonify({
            "success": True,
            "stats": {
                "edges_count": len(edges)
            },
            "edges": edges_json
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/build-user-embeddings', methods=['POST'])
def build_user_embeddings_endpoint():
    """
    유저 임베딩 생성 API (CSV로 저장)
    ---
    tags:
      - Embeddings
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            item_embeddings:
              type: array
              description: 아이템 임베딩 데이터 (content_id, emb_0, emb_1, ...)
              items:
                type: object
            raw_genres:
              type: array
              description: 원본 장르 데이터 (content_id, source, raw_genre_1~3)
              items:
                type: object
            user_preferred_genres:
              type: array
              description: 유저 선호 장르 (user_id, genre, [username])
              items:
                type: object
            contents:
              type: array
              description: 콘텐츠 데이터 (선택사항)
              items:
                type: object
          required:
            - item_embeddings
            - raw_genres
            - user_preferred_genres
    responses:
      200:
        description: 유저 임베딩 생성 성공
        schema:
          properties:
            success:
              type: boolean
              example: true
            message:
              type: string
              example: "유저 임베딩이 성공적으로 생성되었습니다."
            stats:
              type: object
              properties:
                users_count:
                  type: integer
      400:
        description: 잘못된 요청
      500:
        description: 서버 오류
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "error": "JSON 데이터가 필요합니다"}), 400
        
        # 필수 데이터 확인
        if not data.get('item_embeddings'):
            return jsonify({"success": False, "error": "item_embeddings 데이터가 필요합니다"}), 400
        if not data.get('raw_genres'):
            return jsonify({"success": False, "error": "raw_genres 데이터가 필요합니다"}), 400
        if not data.get('user_preferred_genres'):
            return jsonify({"success": False, "error": "user_preferred_genres 데이터가 필요합니다"}), 400
        
        # DataFrame으로 변환
        item_emb_df = pd.DataFrame(data.get('item_embeddings'))
        raw_genres_df = pd.DataFrame(data.get('raw_genres'))
        user_genres_df = pd.DataFrame(data.get('user_preferred_genres'))
        contents_df = pd.DataFrame(data.get('contents')) if data.get('contents') else None
        
        # 유저 임베딩 생성
        print("👤 유저 임베딩 생성 시작...")
        user_embeddings = build_user_embeddings(
            item_emb_df, raw_genres_df, user_genres_df, contents_df
        )
        
        # CSV로 저장
        import os
        OUT_USER = os.path.join(r"csv 데이터\clean", "user_embeddings.csv")
        user_embeddings.to_csv(OUT_USER, index=False, encoding="utf-8-sig")
        print(f"✅ 유저 임베딩 저장: {OUT_USER}")
        
        return jsonify({
            "success": True,
            "message": "유저 임베딩이 성공적으로 생성되었습니다.",
            "stats": {
                "users_count": len(user_embeddings)
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/build-user-embeddings-get-result', methods=['POST'])
def build_user_embeddings_get_result():
    """
    유저 임베딩 생성 후 결과 반환 API (메모리에서)
    ---
    tags:
      - Embeddings
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            item_embeddings:
              type: array
              description: 아이템 임베딩 데이터 (content_id, emb_0, emb_1, ...)
              items:
                type: object
            raw_genres:
              type: array
              description: 원본 장르 데이터 (content_id, source, raw_genre_1~3)
              items:
                type: object
            user_preferred_genres:
              type: array
              description: 유저 선호 장르 (user_id, genre, [username])
              items:
                type: object
            contents:
              type: array
              description: 콘텐츠 데이터 (선택사항)
              items:
                type: object
          required:
            - item_embeddings
            - raw_genres
            - user_preferred_genres
    responses:
      200:
        description: 유저 임베딩 생성 및 결과 조회 성공
        schema:
          properties:
            success:
              type: boolean
              example: true
            stats:
              type: object
              properties:
                users_count:
                  type: integer
            embeddings:
              type: array
              description: 유저 임베딩 데이터 (user_id, username, emb_0, emb_1, ...)
              items:
                type: object
      400:
        description: 잘못된 요청
      500:
        description: 서버 오류
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "error": "JSON 데이터가 필요합니다"}), 400
        
        # 필수 데이터 확인
        if not data.get('item_embeddings'):
            return jsonify({"success": False, "error": "item_embeddings 데이터가 필요합니다"}), 400
        if not data.get('raw_genres'):
            return jsonify({"success": False, "error": "raw_genres 데이터가 필요합니다"}), 400
        if not data.get('user_preferred_genres'):
            return jsonify({"success": False, "error": "user_preferred_genres 데이터가 필요합니다"}), 400
        
        # DataFrame으로 변환
        item_emb_df = pd.DataFrame(data.get('item_embeddings'))
        raw_genres_df = pd.DataFrame(data.get('raw_genres'))
        user_genres_df = pd.DataFrame(data.get('user_preferred_genres'))
        contents_df = pd.DataFrame(data.get('contents')) if data.get('contents') else None
        
        # 유저 임베딩 생성
        user_embeddings = build_user_embeddings(
            item_emb_df, raw_genres_df, user_genres_df, contents_df
        )
        
        # JSON으로 변환
        embeddings_json = user_embeddings.to_dict('records')
        
        return jsonify({
            "success": True,
            "stats": {
                "users_count": len(user_embeddings)
            },
            "embeddings": embeddings_json
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# @app.route('/build-item-embeddings', methods=['POST'])
# def build_item_embeddings_endpoint():
#     """
#     아이템 임베딩 생성 API (Node2Vec + Skip-gram, CSV로 저장)
#     ---
#     tags:
#       - Embeddings
#     parameters:
#       - in: body
#         name: body
#         required: true
#         schema:
#           type: object
#           properties:
#             edges:
#               type: array
#               description: 아이템-아이템 엣지 데이터 (src_content_id, dst_content_id, weight)
#               items:
#                 type: object
#             dim:
#               type: integer
#               description: 임베딩 차원 (기본값: 64)
#               example: 64
#             walk_length:
#               type: integer
#               description: 랜덤 워크 길이 (기본값: 40)
#               example: 40
#             num_walks:
#               type: integer
#               description: 노드당 워크 개수 (기본값: 10)
#               example: 10
#             epochs:
#               type: integer
#               description: 학습 에포크 (기본값: 3)
#               example: 3
#             batch_size:
#               type: integer
#               description: 배치 사이즈 (기본값: 8192)
#               example: 8192
#             lr:
#               type: number
#               description: 학습률 (기본값: 0.025)
#               example: 0.025
#           required:
#             - edges
#     responses:
#       200:
#         description: 아이템 임베딩 생성 성공
#         schema:
#           properties:
#             success:
#               type: boolean
#               example: true
#             message:
#               type: string
#               example: "아이템 임베딩이 성공적으로 생성되었습니다."
#             stats:
#               type: object
#               properties:
#                 items_count:
#                   type: integer
#                 embedding_dim:
#                   type: integer
#       400:
#         description: 잘못된 요청
#       500:
#         description: 서버 오류
#     """
#     try:
#         data = request.get_json()
        
#         if not data:
#             return jsonify({"success": False, "error": "JSON 데이터가 필요합니다"}), 400
        
#         if not data.get('edges'):
#             return jsonify({"success": False, "error": "edges 데이터가 필요합니다"}), 400
        
#         # DataFrame으로 변환
#         edges_df = pd.DataFrame(data.get('edges'))
        
#         # 하이퍼파라미터 (기본값 사용, 요청에서 오버라이드 가능)
#         dim = data.get('dim', 64)
#         walk_length = data.get('walk_length', 40)
#         num_walks = data.get('num_walks', 10)
#         epochs = data.get('epochs', 3)
#         batch_size = data.get('batch_size', 8192)
#         lr = data.get('lr', 0.025)
        
#         # 아이템 임베딩 생성
#         print("🧠 아이템 임베딩 생성 시작...")
#         embeddings = build_item_embeddings(
#             edges_df, 
#             dim=dim, 
#             walk_length=walk_length,
#             num_walks=num_walks,
#             epochs=epochs,
#             batch_size=batch_size,
#             lr=lr
#         )
        
#         # CSV로 저장
#         import os
#         OUT_EMB = os.path.join(r"csv 데이터\clean", "item_embeddings_torch.csv")
#         embeddings.to_csv(OUT_EMB, index=False, encoding="utf-8-sig")
#         print(f"✅ 임베딩 저장: {OUT_EMB}")
        
#         return jsonify({
#             "success": True,
#             "message": "아이템 임베딩이 성공적으로 생성되었습니다.",
#             "stats": {
#                 "items_count": len(embeddings),
#                 "embedding_dim": dim
#             }
#         }), 200
        
#     except Exception as e:
#         return jsonify({
#             "success": False,
#             "error": str(e)
#         }), 500

# @app.route('/build-item-embeddings-get-result', methods=['POST'])
# def build_item_embeddings_get_result():
#     """
#     아이템 임베딩 생성 후 결과 반환 API (Node2Vec + Skip-gram, 메모리에서)
#     ---
#     tags:
#       - Embeddings
#     parameters:
#       - in: body
#         name: body
#         required: true
#         schema:
#           type: object
#           properties:
#             edges:
#               type: array
#               description: 아이템-아이템 엣지 데이터 (src_content_id, dst_content_id, weight)
#               items:
#                 type: object
#             dim:
#               type: integer
#               description: 임베딩 차원 (기본값: 64)
#               example: 64
#             walk_length:
#               type: integer
#               description: 랜덤 워크 길이 (기본값: 40)
#               example: 40
#             num_walks:
#               type: integer
#               description: 노드당 워크 개수 (기본값: 10)
#               example: 10
#             epochs:
#               type: integer
#               description: 학습 에포크 (기본값: 3)
#               example: 3
#             batch_size:
#               type: integer
#               description: 배치 사이즈 (기본값: 8192)
#               example: 8192
#             lr:
#               type: number
#               description: 학습률 (기본값: 0.025)
#               example: 0.025
#           required:
#             - edges
#     responses:
#       200:
#         description: 아이템 임베딩 생성 및 결과 조회 성공
#         schema:
#           properties:
#             success:
#               type: boolean
#               example: true
#             stats:
#               type: object
#               properties:
#                 items_count:
#                   type: integer
#                 embedding_dim:
#                   type: integer
#             embeddings:
#               type: array
#               description: 아이템 임베딩 데이터 (content_id, emb_0, emb_1, ...)
#               items:
#                 type: object
#       400:
#         description: 잘못된 요청
#       500:
#         description: 서버 오류
#     """
#     try:
#         data = request.get_json()
        
#         if not data:
#             return jsonify({"success": False, "error": "JSON 데이터가 필요합니다"}), 400
        
#         if not data.get('edges'):
#             return jsonify({"success": False, "error": "edges 데이터가 필요합니다"}), 400
        
#         # DataFrame으로 변환
#         edges_df = pd.DataFrame(data.get('edges'))
        
#         # 하이퍼파라미터 (기본값 사용, 요청에서 오버라이드 가능)
#         dim = data.get('dim', 64)
#         walk_length = data.get('walk_length', 40)
#         num_walks = data.get('num_walks', 10)
#         epochs = data.get('epochs', 3)
#         batch_size = data.get('batch_size', 8192)
#         lr = data.get('lr', 0.025)
        
#         # 아이템 임베딩 생성
#         embeddings = build_item_embeddings(
#             edges_df, 
#             dim=dim, 
#             walk_length=walk_length,
#             num_walks=num_walks,
#             epochs=epochs,
#             batch_size=batch_size,
#             lr=lr
#         )
        
#         # JSON으로 변환
#         embeddings_json = embeddings.to_dict('records')
        
#         return jsonify({
#             "success": True,
#             "stats": {
#                 "items_count": len(embeddings),
#                 "embedding_dim": dim
#             },
#             "embeddings": embeddings_json
#         }), 200
        
#     except Exception as e:
#         return jsonify({
#             "success": False,
#             "error": str(e)
#         }), 500

if __name__ == '__main__':
    # USE_API_MODE를 True로 설정하려면 make_bigraph.py에서 수정해야 함
    app.run(debug=True, host='0.0.0.0', port=5000)
