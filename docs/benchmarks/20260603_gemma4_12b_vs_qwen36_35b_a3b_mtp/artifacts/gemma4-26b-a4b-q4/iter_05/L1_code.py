from collections import deque
    from typing import Any, Dict, Optional

    class JobQueue:
        def __init__(self):
            self._queue = deque()
            self._jobs = {}
            self._results = {}

        def add_job(self, job_id: str, data: dict) -> str:
            self._jobs[job_id] = data
            self._queue.append(job_id)
            return job_id

        def process_next_job(self, worker_func):
            if not self._queue:
                return None
            job_id = self._queue.popleft()
            data = self._jobs.pop(job_id)
            result = worker_func(data)
            self._results[job_id] = result
            return result

        def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
            return self._results.get(job_id)