from collections import deque
from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._data = {}
        self._retry_count = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> None:
        if job_id in self._queue:
            self._retry_count[job_id] = self._retry_count.get(job_id, 0) + 1
            return
        self._queue.append(job_id)
        self._data[job_id] = data
        self._retry_count[job_id] = 0

    def process_job(self, job_id: str, processor: Callable[[Dict[str, Any]], Any]) -> bool:
        data = self._data.pop(job_id, None)
        if data is None:
            raise KeyError(f"Job {job_id} not found")
        fail_count = self._retry_count.get(job_id, 0)
        for attempt in range(1, 5):
            if attempt > 1:
                delay = 2 ** (attempt - 2)
                data['delay'] = delay
            try:
                result = processor(data)
                return True
            except Exception:
                fail_count += 1
                self._retry_count[job_id] = fail_count
                if attempt == 4:
                    break
        return False