from collections import deque
from typing import Callable, Dict, Optional

class JobQueue:
    def __init__(self) -> None:
        self._queue = deque()
        self._results: Dict[str, Dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append((job_id, data))
        return job_id

    def get_result(self, job_id: str) -> Optional[Dict]:
        # Return immediate result if already processed
        if job_id in self._results:
            return self._results[job_id]

        # Process jobs in FIFO order until the requested one is completed
        while self._queue:
            current_job_id, current_data = self._queue.popleft()

            # Process the current job
            if 'function' in current_data and isinstance(current_data['function'], Callable):
                processed = current_data['function']()
            else:
                processed = current_data.get('task', 'No task provided')

            # Store the result for this job
            self._results[current_job_id] = {
                'job_id': current_job_id,
                'result': processed
            }

            # If this is the requested job, return its result
            if current_job_id == job_id:
                return self._results[current_job_id]

        # Job not found (e.g., never added)
        return None