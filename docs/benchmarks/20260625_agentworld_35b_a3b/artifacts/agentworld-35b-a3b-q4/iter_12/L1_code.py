import collections
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        self.job_queue = collections.deque()
        self.job_data: Dict[str, dict] = {}
        self.job_results: Dict[str, Optional[dict]] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        if job_id not in self.job_data:
            self.job_queue.append(job_id)
            self.job_data[job_id] = data
            self.job_results[job_id] = None
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        if job_id in self.job_results and self.job_results[job_id] is not None:
            return self.job_results[job_id]
        return None

    def set_result(self, job_id: str, result: dict) -> None:
        if job_id in self.job_results:
            self.job_results[job_id] = result
        else:
            raise ValueError(f"Job {job_id} not found in queue")