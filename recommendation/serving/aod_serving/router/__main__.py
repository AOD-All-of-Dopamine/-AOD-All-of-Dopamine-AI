"""python -m aod_serving.router"""
import logging, os
import uvicorn

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run("aod_serving.router.app:app_from_env", factory=True, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
