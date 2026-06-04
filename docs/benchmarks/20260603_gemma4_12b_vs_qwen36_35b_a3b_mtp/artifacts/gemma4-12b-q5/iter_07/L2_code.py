from typing import Callable, Any, Dict

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        if "retries" not in data:
            data["retries"] = 0
        if "delays" not in data:
            data["delays"] = []
        self.jobs[job_id] = data

    def process_job(self, job_id: str, Processor: Callable) -> bool:
        job_data = self.jobs.get(job_id)
        if job_data is None:
            return False

        for attempt in range(4):
            try:
                Processor(job_data)
                return True
            except Exception:
                if attempt < 3:
                    delay = 2 ** attempt
                    job_data["retries"] += 1
                    job_data["delays"].append(delay))
                else:
                    return False
        return False