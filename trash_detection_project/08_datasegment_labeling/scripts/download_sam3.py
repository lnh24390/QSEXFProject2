"""SAM 3 / 3.1 다운로드 도우미 (HuggingFace 게이트 저장소 전용).

`facebook/sam3` 와 `facebook/sam3.1` 은 수동 승인이 필요한 게이트 저장소라
익명 다운로드가 401 로 막힙니다. 이 스크립트는 토큰을 받아 접근 권한을 먼저
확인한 뒤, 프로젝트 다운로더로 `weights/<계열>/<이름>/` 에 내려받습니다.

    python scripts/download_sam3.py                  # 토큰을 숨김 입력으로 물어봄
    python scripts/download_sam3.py --only sam3      # 하나만
    python scripts/download_sam3.py --save-token     # 다음부터 다시 안 묻게 저장
    HF_TOKEN=hf_xxx python scripts/download_sam3.py  # 환경변수로 넘기기

토큰은 화면에 표시되지 않고, 로그에도 남기지 않습니다.
"""
from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dsl.sam import downloader                       # noqa: E402
from dsl.sam.catalog import CATALOG                  # noqa: E402

GATED = ["sam3", "sam3.1"]
ACCESS_URL = "https://huggingface.co/{repo}"


def _mask(token: str) -> str:
    return f"{token[:6]}…{token[-4:]}" if len(token) > 12 else "…"


def get_token(save: bool) -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        print("HF_TOKEN 환경변수를 사용합니다.")
    else:
        try:
            from huggingface_hub import get_token as hub_token
            token = hub_token()
        except ImportError:
            token = None
        if token:
            print("이미 로그인된 토큰을 사용합니다.")
        else:
            print("HuggingFace Read 토큰을 붙여넣으세요 "
                  "(https://huggingface.co/settings/tokens).")
            print("입력은 화면에 표시되지 않습니다.")
            token = getpass.getpass("token:").strip()
    if not token:
        raise SystemExit("토큰이 비어 있습니다.")

    if save:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        print("토큰을 저장했습니다 (~/.cache/huggingface/token).")
    else:
        # 이 프로세스와 자식 프로세스에서만 쓰이도록
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    return token


def check_access(repo: str, token: str) -> bool:
    """승인 대기 중인지, 토큰이 틀렸는지 구분해서 알려준다.

    `model_info` 는 게이트 저장소에서도 익명으로 통과하므로(메타데이터는 공개)
    권한 확인이 되지 않는다. 실제로 게이트가 걸리는 파일 다운로드로 확인한다.
    """
    import tempfile

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import (GatedRepoError, RepositoryNotFoundError,
                                       HfHubHTTPError)
    try:
        hf_hub_download(repo, "config.json", token=token,
                        local_dir=tempfile.mkdtemp())
        return True
    except GatedRepoError:
        print(f"  ✗ {repo}: 아직 접근 승인이 나지 않았습니다.")
        print(f"    {ACCESS_URL.format(repo=repo)} 에서 라이선스에 동의하고 "
              "승인될 때까지 기다린 뒤 다시 실행하세요.")
    except RepositoryNotFoundError:
        print(f"  ✗ {repo}: 저장소를 찾을 수 없습니다(이름 변경/삭제). "
              "모델 관리자의 [URL/저장소 수정]으로 바꿀 수 있습니다.")
    except HfHubHTTPError as e:
        code = getattr(e.response, "status_code", "?")
        if code == 401:
            print(f"  ✗ {repo}: 토큰이 유효하지 않습니다(401). Read 권한 토큰인지 "
                  "확인하세요.")
        else:
            print(f"  ✗ {repo}: HTTP {code} — {str(e).splitlines()[0][:120]}")
    except Exception as e:                    # 네트워크/프록시/형식 오류 등
        print(f"  ✗ {repo}: 확인 실패 — {type(e).__name__}: "
              f"{str(e).splitlines()[0][:120]}")
    return False


#: 저장소에서 받아야 하는 파일(가중치 + transformers 설정). 확장자 기준.
WANTED_SUFFIXES = (".safetensors", ".pt", ".pth", ".bin",
                   ".json", ".txt", ".model", ".yaml", ".py")


def missing_files(spec, weights_dir: Path, token: str) -> list[str]:
    """저장소에는 있는데 로컬에 없는 파일 목록. 부분 다운로드를 잡아낸다."""
    from huggingface_hub import HfApi
    try:
        info = HfApi().model_info(spec.repo_id, token=token)
    except Exception:
        return []                     # 목록을 못 받으면 판단 보류
    local = spec.local_path(weights_dir)
    out = []
    for s in info.siblings or []:
        name = s.rfilename
        if not name.endswith(WANTED_SUFFIXES) or name.startswith("."):
            continue
        if not (local / name).is_file():
            out.append(name)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="SAM 3 / 3.1 다운로드")
    ap.add_argument("--only", choices=GATED, help="하나만 받기")
    ap.add_argument("--save-token", action="store_true",
                    help="토큰을 저장해 다음부터 묻지 않기")
    ap.add_argument("--weights-dir", default=str(ROOT / "weights"))
    args = ap.parse_args()

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        raise SystemExit("huggingface_hub 가 필요합니다.\n"
                         "    uv pip install huggingface_hub")

    keys = [args.only] if args.only else GATED
    token = get_token(args.save_token)
    print(f"토큰 {_mask(token)} 로 접근 권한을 확인합니다.\n")

    wd = Path(args.weights_dir)
    ok = []
    for key in keys:
        spec = CATALOG[key]
        print(f"[{key}] {spec.display}  ({spec.repo_id}, 약 {spec.size_mb}MB)")
        missing = missing_files(spec, wd, token)
        if spec.is_installed(wd) and not missing:
            print(f"  · 이미 설치됨: {spec.local_path(wd)}")
            ok.append(key)
            continue
        if missing:
            print(f"  · 빠진 파일 {len(missing)}개: "
                  + ", ".join(missing[:4])
                  + (" …" if len(missing) > 4 else ""))
        if not check_access(spec.repo_id, token):
            continue
        print("  · 접근 확인. 내려받는 중…")
        try:
            dest = downloader.install(
                key, wd,
                progress=lambda done, total, label=key: print(
                    f"\r    {label}: {done}/{total}", end="", flush=True),
                force=bool(missing))       # 부분 다운로드를 마저 채운다
            print(f"\r  ✓ 완료: {dest}" + " " * 20)
            ok.append(key)
        except downloader.DownloadError as e:
            print(f"\r  ✗ 실패: {e}")

    print("\n=== 결과 ===")
    for key in keys:
        spec = CATALOG[key]
        state = "설치됨" if spec.is_installed(wd) else "미설치"
        print(f"  {key:8s} {state:6s} {spec.local_path(wd)}")
    if len(ok) == len(keys):
        print("\n앱에서 [SAM] 탭 → 모델 목록에 SAM 3 가 ✔ 로 뜹니다.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
