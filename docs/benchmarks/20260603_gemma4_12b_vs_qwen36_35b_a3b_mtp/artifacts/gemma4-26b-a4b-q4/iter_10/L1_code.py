from collections import deque
    from typing import Dict, Any, Optional

    class JobQueue:
        def __init__(self):
            self._queue = deque()
            self._results = {}
            self._jobs = {}

        def add_job(self, job_id: str, data: dict) -> str:
            self._queue.append(job_id)
            self._jobs[job_id] = data
            return job_id

        def process_next_job(self):
            if not self._queue:
                return
            
            job_id = self._queue.popleft()
            data = self._jobs.pop(job_id)
            
            # Simulate processing
            result = {"processed": True, "original_data": data}
            self._results[job_id] = result

        def get_result(self, job_id: str) -> Optional[dict]:
            return self._results.get(job_id)