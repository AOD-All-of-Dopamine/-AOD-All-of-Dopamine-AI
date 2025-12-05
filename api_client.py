# -*- coding: utf-8 -*-

import requests
import json
import pandas as pd

# API 서버 URL
API_URL = "http://localhost:5000"

def load_csv_data(file_path):
    """CSV 파일을 읽어서 리스트 형태로 변환"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        return df.to_dict('records')
    except Exception as e:
        print(f"⚠️ 파일 읽기 실패 ({file_path}): {e}")
        return None

def call_build_graph_api(contents_data, av_data=None, game_data=None, webnovel_data=None, raw_item_data=None):
    """
    /build-graph API 호출 (결과를 CSV로 저장)
    """
    payload = {
        "contents": contents_data,
        "av": av_data,
        "game": game_data,
        "webnovel": webnovel_data,
        "raw_item": raw_item_data
    }
    
    try:
        response = requests.post(f"{API_URL}/build-graph", json=payload)
        result = response.json()
        
        if result.get("success"):
            print("✅ 그래프 생성 성공!")
            print(f"   - 콘텐츠 노드: {result['stats']['content_nodes_count']}")
            print(f"   - 메타 노드: {result['stats']['meta_nodes_count']}")
            print(f"   - 엣지: {result['stats']['edges_count']}")
        else:
            print(f"❌ API 에러: {result.get('error')}")
        
        return result
    except Exception as e:
        print(f"❌ API 호출 실패: {e}")
        return None

def call_build_graph_get_result_api(contents_data, av_data=None, game_data=None, webnovel_data=None, raw_item_data=None):
    """
    /build-graph-get-result API 호출 (결과를 메모리에서 반환)
    """
    payload = {
        "contents": contents_data,
        "av": av_data,
        "game": game_data,
        "webnovel": webnovel_data,
        "raw_item": raw_item_data
    }
    
    try:
        response = requests.post(f"{API_URL}/build-graph-get-result", json=payload)
        result = response.json()
        
        if result.get("success"):
            print("✅ 그래프 생성 및 조회 성공!")
            print(f"   - 콘텐츠 노드: {result['stats']['content_nodes_count']}")
            print(f"   - 메타 노드: {result['stats']['meta_nodes_count']}")
            print(f"   - 엣지: {result['stats']['edges_count']}")
            
            # 반환된 데이터로 DataFrame 생성 가능
            nodes_df = pd.DataFrame(result.get('nodes', []))
            meta_df = pd.DataFrame(result.get('meta_nodes', []))
            edges_df = pd.DataFrame(result.get('edges', []))
            
            return {
                "success": True,
                "nodes": nodes_df,
                "meta_nodes": meta_df,
                "edges": edges_df,
                "stats": result.get('stats')
            }
        else:
            print(f"❌ API 에러: {result.get('error')}")
            return {"success": False}
        
    except Exception as e:
        print(f"❌ API 호출 실패: {e}")
        return {"success": False}

if __name__ == "__main__":
    """
    사용 예시:
    
    # 1. CSV 파일에서 데이터 로드
    contents_data = load_csv_data("csv 데이터/clean/contents.csv")
    av_data = load_csv_data("csv 데이터/clean/av_contents.csv")
    
    # 2. API 호출 (결과를 CSV로 저장)
    result = call_build_graph_api(contents_data, av_data=av_data)
    
    # 또는 3. API 호출 (결과를 메모리에서 조회)
    result = call_build_graph_get_result_api(contents_data, av_data=av_data)
    if result["success"]:
        nodes_df = result["nodes"]
        meta_df = result["meta_nodes"]
        edges_df = result["edges"]
    """
    print("API 클라이언트 모듈 로드 완료")
    print("call_build_graph_api() 또는 call_build_graph_get_result_api() 함수를 사용하세요")
