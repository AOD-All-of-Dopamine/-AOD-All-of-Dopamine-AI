"""python -m aod_serving.engine — 환경변수로 설정한다(app.py 머리말 참고)."""
import logging, os
import uvicorn

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run("aod_serving.engine.app:app_from_env", factory=True, host="0.0.0.0",
                port=int(os.environ.get("PORT", "8000")), workers=int(os.environ.get("WORKERS", "1")))
