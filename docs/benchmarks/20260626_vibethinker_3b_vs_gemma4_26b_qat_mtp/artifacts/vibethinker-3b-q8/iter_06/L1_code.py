from typing import Dict, List, Tuple, Optional

class JobQueue:
    def __init__(self) -> None:
        self._jobs: List[Tuple[str, Dict]] = []
        self._next_idx: int = 0
        self._results: Dict[str, Dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._jobs.append((job_id, data))
        return job_id

    def get_result(self, job_id: str) -> Optional[Dict]:
        target_idx = None
        for i, (jid, _) in enumerate(self._jobs):
            if jid == job_id:
                target_idx = i
                break
        if target_idx is None:
            return None

        for i in range(self._next_idx, target_idx + 1):
            jid, data = self._jobs[i]
            self._results[jid] = data.copy()

        self._next_idx = target_idx + 1
        return self._results[job_id]