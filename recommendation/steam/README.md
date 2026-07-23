# AOD Steam Recommendation — S1 Semantic Retrieval 실험

spec: ../../docs/superpowers/specs/2026-07-21-steam-s1-experiment-spec.md

## Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

## Pipeline 순서
python -m src.data_loader
python -m src.text_builder
python -m src.anchor_builder generate
# (사람) anchor_candidates_80.xlsx 의 include_YN 을 정확히 40개 Y로
python -m src.anchor_builder finalize
python -m src.tfidf_baseline
python -m src.embed_qwen
python -m src.retrieve
python -m src.export_evaluation            # full
python -m src.export_evaluation --pilot    # pilot
# (사람) evaluation.xlsx Judgments 시트 채점
python -m src.validate_evaluation fill artifacts/s1/evaluation.xlsx
python -m src.validate_evaluation check artifacts/s1/evaluation.xlsx
python -m src.evaluate
python -m src.report
