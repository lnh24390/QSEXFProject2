#!/usr/bin/env bash
# 이 저장소의 커밋된 내용을 QSEXFProject2 의
# trash_detection_project/06_trash_detection_app/ 으로 반영하고 푸시한다.
#
#   tool/sync_to_qsexf.sh                  # 마지막 커밋 제목을 그대로 설명으로 사용
#   tool/sync_to_qsexf.sh "사진 모드 수정"    # 설명 직접 지정
#
# 대상 저장소 위치는 QSEXF_DIR 환경변수로 바꿀 수 있다.
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${QSEXF_DIR:-$HOME/Desktop/QSEXFProject2/QSEXFProject2}"
DEST="$TARGET/trash_detection_project/06_trash_detection_app"
REL="trash_detection_project/06_trash_detection_app"

if [ ! -d "$TARGET/.git" ]; then
  echo "error: QSEXFProject2 클론을 찾지 못했습니다 -> $TARGET" >&2
  echo "       QSEXF_DIR 로 경로를 지정하세요." >&2
  exit 1
fi

MSG="${1:-$(git -C "$SRC" log -1 --format=%s)}"

# 커밋된 파일만 복사한다. 작업 중인 파일이나 빌드 산출물이 섞이지 않는다.
rm -rf "$DEST"
mkdir -p "$DEST"
git -C "$SRC" ls-files -z | while IFS= read -r -d '' f; do
  mkdir -p "$DEST/$(dirname "$f")"
  cp "$SRC/$f" "$DEST/$f"
done

# 상위 trash_detection_project/.gitignore 가 *.tflite / 이미지들을 막아
# 앱 필수 파일이 빠진다. 이 폴더에서만 다시 허용한다.
{
  echo "# 상위 trash_detection_project/.gitignore 가 막는 앱 필수 파일을 이 폴더에서는 허용한다"
  echo '!*.tflite'
  echo '!*.png'
  echo '!*.jpg'
  echo '!*.jpeg'
  echo ""
  cat "$DEST/.gitignore"
} > "$DEST/.gitignore.new"
mv "$DEST/.gitignore.new" "$DEST/.gitignore"

cd "$TARGET"
# 팀원 작업이 섞이지 않도록 06 폴더만 스테이징한다.
git add -- "$REL"
if git diff --cached --quiet -- "$REL"; then
  echo ">> 바뀐 내용 없음 — 건너뜁니다."
  exit 0
fi

git commit -q -m "06_trash_detection_app 동기화: $MSG" -- "$REL"
echo ">> 커밋: $(git log -1 --format='%h %s')"

# 다른 작업 중인 파일이 있어도 리베이스가 멈추지 않도록 --autostash.
git pull --rebase --autostash -q origin main
git push -q origin main
echo ">> 푸시 완료 -> lnh24390/QSEXFProject2 ($REL)"
