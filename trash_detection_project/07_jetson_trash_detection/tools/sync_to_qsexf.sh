#!/usr/bin/env bash
# 이 저장소의 커밋된 내용을 QSEXFProject2 의
# trash_detection_project/07_jetson_trash_detection/ 으로 반영하고 푸시한다.
#
#   tools/sync_to_qsexf.sh                # 마지막 커밋 제목을 그대로 설명으로 사용
#   tools/sync_to_qsexf.sh "노출 조절 추가"  # 설명 직접 지정
#
# 대상 저장소 위치는 QSEXF_DIR 환경변수로 바꿀 수 있다.
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${QSEXF_DIR:-$HOME/Desktop/QSEXFProject2/QSEXFProject2}"
DEST="$TARGET/trash_detection_project/07_jetson_trash_detection"
REL="trash_detection_project/07_jetson_trash_detection"

if [ ! -d "$TARGET/.git" ]; then
  echo "error: QSEXFProject2 클론을 찾지 못했습니다 -> $TARGET" >&2
  echo "       QSEXF_DIR 로 경로를 지정하세요." >&2
  exit 1
fi

MSG="${1:-$(git -C "$SRC" log -1 --format=%s)}"

# 커밋된 파일만 복사한다. 작업 중인 파일이나 output/ 같은 산출물이 섞이지 않는다.
#   .github/       : 대상 저장소에서 워크플로를 돌릴 이유가 없다
#   model/*.pt     : 약 158MB. 상위 trash_detection_project/.gitignore 가 어차피
#                    막고 있고, 가중치는 원본 저장소에서 받는다
rm -rf "$DEST"
mkdir -p "$DEST"
git -C "$SRC" ls-files -z | while IFS= read -r -d '' f; do
  case "$f" in
    .github/*|model/*.pt) continue ;;
  esac
  mkdir -p "$DEST/$(dirname "$f")"
  cp "$SRC/$f" "$DEST/$f"
done

# 상위 .gitignore 가 *.jpg / *.png 를 막아 README 의 데모 이미지가 빠진다.
# 문서용 이미지만 이 폴더에서 다시 허용한다.
{
  echo "# 상위 trash_detection_project/.gitignore 가 막는 문서용 이미지를 이 폴더에서는 허용한다"
  echo '!docs/*.jpg'
  echo '!docs/*.png'
  echo ""
  cat "$DEST/.gitignore"
} > "$DEST/.gitignore.new"
mv "$DEST/.gitignore.new" "$DEST/.gitignore"

cd "$TARGET"
# 팀원 작업이 섞이지 않도록 07 폴더만 스테이징한다.
git add -- "$REL"
if git diff --cached --quiet -- "$REL"; then
  echo ">> 바뀐 내용 없음 — 건너뜁니다."
  exit 0
fi

git commit -q -m "07_jetson_trash_detection 동기화: $MSG" -- "$REL"
echo ">> 커밋: $(git log -1 --format='%h %s')"

# 다른 작업 중인 파일이 있어도 리베이스가 멈추지 않도록 --autostash.
git pull --rebase --autostash -q origin main
git push -q origin main
echo ">> 푸시 완료 -> lnh24390/QSEXFProject2 ($REL)"
