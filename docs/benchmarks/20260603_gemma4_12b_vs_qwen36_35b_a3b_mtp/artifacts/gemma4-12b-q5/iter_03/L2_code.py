from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        self.jobs[job_id] = data

    def process_job(self, job_id: str, Processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        job_data = self.jobs[job_id]

        if "retry_count" not in job_data:
            job_data["retry_count"] = 0
        if "delays" not in job_data:
            job_data["delays"] = []

        delays = [1, 2, 4]

        for attempt in range(4):
            try:
                Processor(job_data)
                return True
            except Exception:
                if attempt < 3:
                    job_data["retry_count"] += 1
                    job_data["delays"].append(delays[attempt]))
                else:
                    return False
        return False