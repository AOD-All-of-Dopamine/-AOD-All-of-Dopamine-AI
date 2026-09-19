"""고정 파일 세 개가 같은 버전을 가리키는지 — 엔진·라우터·테스트 환경이 어긋나면 동일성 테스트의 의미가 없다."""
from pathlib import Path

REQ = Path(__file__).resolve().parents[1] / "requirements"


def pins(name: str) -> dict[str, str]:
    out = {}
    for line in (REQ / name).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "==" in line:
            pkg, ver = line.split("==", 1)
            out[pkg.lower().replace("_", "-")] = ver
    return out


def test_numeric_stack_matches_eval_server():
    s = pins("serving.txt")
    assert (s["numpy"], s["pandas"], s["pyarrow"], s["pyyaml"]) == ("2.5.1", "3.0.5", "25.0.0", "6.0.3")


def test_router_has_no_numeric_stack():
    assert not {"numpy", "pandas", "pyarrow"} & set(pins("router.txt"))


def test_dev_is_superset_with_same_versions():
    dev = pins("dev.txt")
    for name in ("serving.txt", "router.txt"):
        for pkg, ver in pins(name).items():
            assert dev.get(pkg) == ver, f"{name}: {pkg}=={ver} 가 dev.txt({dev.get(pkg)}) 와 다르다"
