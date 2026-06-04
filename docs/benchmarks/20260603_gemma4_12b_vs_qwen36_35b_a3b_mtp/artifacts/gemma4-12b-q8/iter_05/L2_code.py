from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: dict):
        self.jobs[job_id] = {
            "data": data,
            "retries": 0,
            "delay_history": []
        }

    def process_job(self, job_id: str, processor: Callable[[dict], Any]) -> bool:
        if job_id not in self.jobs:
            return False

        job = self.jobs[job_id]
        data = job["data"]

        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 3:
                    delay = 2 ** job["retries"]
                    job["delay_history"].append(delay)
                    job["retries"] += 1
                else:
                    break

        return False