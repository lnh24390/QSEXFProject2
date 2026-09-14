"""평가 캐시를 결과 폴더 옆에 두고, 내용이 바뀌면 무효화한다."""
from pathlib import Path
from ultralytics.data.dataset import YOLODataset


class EvaluationDataset(YOLODataset):
    def get_cache_hash(self):
        return self.data['_analysis_fingerprint']

    def _load_or_scan_cache(self, cache_path, cache_hash):
        local_cache = Path(self.data['_analysis_cache_dir'])/'labels.cache'
        return super()._load_or_scan_cache(local_cache, cache_hash)
