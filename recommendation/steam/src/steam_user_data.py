import os

import requests

from src.config import resolve_path

API_BASE = "https://api.steampowered.com"
# 백엔드 리포의 .env 를 그대로 읽는다. 경로는 환경변수로 주입한다(.env.example 참고).
STEAM_API_KEY_ENV_FILE = "${AOD_BACK_ROOT}/.env"


def _load_api_key() -> str:
    """STEAM_API_KEY 를 환경변수에서, 없으면 백엔드 .env 파일에서 읽는다."""
    if key := os.environ.get("STEAM_API_KEY"):
        return key
    path = resolve_path(STEAM_API_KEY_ENV_FILE)
    if not path.exists():
        raise FileNotFoundError(
            f"STEAM_API_KEY 환경변수도 없고 {path} 도 없습니다. "
            f".env.example 을 참고해 STEAM_API_KEY 또는 AOD_BACK_ROOT 를 설정하세요."
        )
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("STEAM_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise KeyError(f"{path} 안에 STEAM_API_KEY 가 없습니다")


def fetch_owned_games(steam_id: str) -> list[dict]:
    key = _load_api_key()
    url = f"{API_BASE}/IPlayerService/GetOwnedGames/v1/"
    params = {"key": key, "steamid": steam_id, "format": "json", "include_appinfo": True}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", {}).get("games", [])


def fetch_player_summary(steam_id: str) -> dict | None:
    key = _load_api_key()
    url = f"{API_BASE}/ISteamUser/GetPlayerSummaries/v2/"
    params = {"key": key, "steamids": steam_id, "format": "json"}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    players = data.get("response", {}).get("players", [])
    return players[0] if players else None


def format_user_data(steam_id: str) -> dict:
    summary = fetch_player_summary(steam_id)
    games = fetch_owned_games(steam_id)
    game_list = []
    for g in sorted(games, key=lambda x: x.get("playtime_forever", 0), reverse=True):
        game_list.append({
            "steam_appid": g["appid"],
            "name": g.get("name", ""),
            "playtime_hours": round(g.get("playtime_forever", 0) / 60, 1),
        })
    return {
        "steam_id": steam_id,
        "persona_name": summary.get("personaname", "") if summary else "",
        "total_games": len(games),
        "total_playtime_hours": round(sum(g.get("playtime_forever", 0) for g in games) / 60, 1),
        "games": game_list,
    }
