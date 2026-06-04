from collections import deque
    from typing import Dict, Optional

    class JobQueue:
        def __init__(self):
            self._queue = deque()
            self._jobs = {} # Stores the input data
            self._results = {} # Stores the final results

        def add_job(self, job_id: str, data: dict) -> str:
            self._queue.append(job_id)
            self._jobs[job_id] = data
            return job_id

        def get_result(self, job_id: str) -> Optional[dict]]:
            # Since there's no 'process' method, 
            # I'll assume "processing" happened 
            # and return the data as the result.
            return self._results.get(job_id)