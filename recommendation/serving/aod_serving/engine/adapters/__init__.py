from aod_serving.engine.adapters.steam import SteamAdapter
from aod_serving.engine.adapters.tmdb import TmdbAdapter
from aod_serving.engine.adapters.webtoon import WebtoonAdapter
from aod_serving.engine.adapters.webnovel import WebnovelAdapter

ADAPTERS = {a.platform: a for a in (SteamAdapter, TmdbAdapter, WebtoonAdapter, WebnovelAdapter)}
