from typing import Callable

class JobQueue:
    def __init__(self):
        self.jobs = {}

    def add_job(self, job_id: str, data: dict):
        # Initialize retry tracking within the data dictionary
        data["retry_count"] = 0
        data["delays"] = []
        self.jobs[job_id] = data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]

        for attempt in range(4):
            try:
                processor(data)
                return True
            except Exception:
                if attempt < 3:
                    # Calculate exponential backoff: 2^0=1, 2^1=2, 2^2=4
                    delay = 2 ** attempt
                    data["retry_count"] += 1
                    data["delays"].append(delay)
                else:
                    # Final attempt failed
                    break
        return False