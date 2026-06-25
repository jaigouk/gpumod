from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        self._jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self._jobs:
            return False

        data = self._jobs[job_id]
        delays = []
        retry_count = 0

        for _ in range(4):
            try:
                processor(data)
                return True
            except Exception:
                retry_count += 1
                if retry_count <= 3:
                    delays.append(2 ** (retry_count - 1))

        data['delays'] = delays
        data['retry_count'] = retry_count
        return False