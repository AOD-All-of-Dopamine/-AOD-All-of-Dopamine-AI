import os
import requests

STEAM_API_KEY_PATH = os.path.expanduser(
    "~/projects/-AOD-All-of-Dopamine-back/.env"
)
API_BASE = "https://api.steampowered.com"


def _load_api_key() -> str:
    if not os.path.exists(STEAM_API_KEY_PATH):
        raise FileNotFoundError(
            f"Steam API key not found at {STEAM_API_KEY_PATH}"
        )
    with open(STEAM_API_KEY_PATH) as f:
        for line in f:
            if line.startswith("STEAM_API_KEY="):
                return line.strip().split("=", 1)[1]
    raise KeyError("STEAM_API_KEY not found in .env")


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
