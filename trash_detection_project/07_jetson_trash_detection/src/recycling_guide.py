"""지자체별 분리수거 안내.

trash_sorter_2026-09-15 (Flutter 앱)의 assets/config/recycling_guide.json 을
그대로 가져와 쓴다 — 카테고리·클래스 매핑·지역 규칙 모두 같은 JSON 하나가
기준(source of truth)이다. 두 프로젝트가 따로 유지보수되지 않도록, 모델 클래스가
바뀌면 이 JSON(config/recycling_guide.json)만 고치면 된다(Dart 앱과 동일한 원칙).

앱은 GPS로 지역을 자동 판별하지만 Jetson 데스크톱에는 GPS가 없다. 대신
config.json 의 "region" 값을 수동으로 골라 쓴다('g' 키로 순환).
"""

import json
import re
from pathlib import Path


def normalize_class_name(name):
    """앱(RecyclingGuide.normalizeClassName)과 동일한 정규화 규칙."""
    s = str(name).strip().lower()
    s = re.sub(r"[\s\-&/]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


class RecyclingGuide:
    """클래스명 -> 분리수거 카테고리 -> (지역별) 버릴 곳/방법."""

    def __init__(self, path):
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self.categories = data.get("categories", {})
        self.classes = {normalize_class_name(k): v for k, v in data.get("classes", {}).items()}
        self.rules = data.get("rules", [])
        self.ignore = {normalize_class_name(n) for n in data.get("ignore", [])}
        self.fallback = data.get("fallback", "general")
        self.regions = data.get("regions", {})
        if "default" not in self.regions:
            self.regions["default"] = {"name": "공통(전국 기준)", "notice": "", "categories": {}}

    # ------------------------------------------------------------ 지역

    def region_ids(self):
        """default 를 맨 앞에 두고 나머지는 JSON 순서대로."""
        ids = [r for r in self.regions if r != "default"]
        return ["default", *ids]

    def region_name(self, region_id):
        return self.regions.get(region_id, self.regions["default"]).get("name", region_id)

    def next_region(self, region_id):
        ids = self.region_ids()
        try:
            i = ids.index(region_id)
        except ValueError:
            i = -1
        return ids[(i + 1) % len(ids)]

    # ---------------------------------------------------------- 매핑

    def resolve_category_id(self, class_name):
        """클래스명 -> 카테고리 id. 무시 대상이면 None."""
        key = normalize_class_name(class_name)
        if key in self.ignore:
            return None
        if key in self.classes:
            return self.classes[key]
        for rule in self.rules:
            if all(kw in key for kw in rule["contains"]):
                return rule["category"]
        return self.fallback

    def lookup(self, class_name, region_id="default"):
        """클래스 하나를 해당 지역 기준 안내로 변환한다. 무시 대상이면 None."""
        cat_id = self.resolve_category_id(class_name)
        if cat_id is None:
            return None

        base = self.categories.get(cat_id) or self.categories.get(self.fallback, {})
        region = self.regions.get(region_id, self.regions["default"])
        override = region.get("categories", {}).get(cat_id, {})

        return {
            "category": cat_id,
            "name": base.get("name", cat_id),
            "bin": base.get("bin", ""),
            "instruction": override.get("instruction") or base.get("instruction", ""),
            "color": base.get("color", "#607D8B"),
        }

    def notice(self, region_id="default"):
        return self.regions.get(region_id, self.regions["default"]).get("notice", "")

    def summarize(self, counts, region_id="default", max_items=3):
        """Detector.summarize() 가 준 {클래스명: 개수} 를 화면에 띄울 안내 목록으로.

        개수(=화면에 잡힌 확신도 우선순위) 순서를 그대로 따르고, 같은 카테고리는
        한 번만 보여준다(페트병 3개가 잡혀도 줄은 하나).
        """
        items = []
        seen = set()
        for class_name in counts:  # counts 는 이미 개수 내림차순
            info = self.lookup(class_name, region_id)
            if info is None or info["category"] in seen:
                continue
            seen.add(info["category"])
            items.append(info)
            if len(items) >= max_items:
                break
        return items
