from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: dict):
        # Store the provided data dictionary and initialize retry tracking metadata
        self.jobs[job_id] = data
        self.jobs[job_id]["__retries__"] = 0
        self.jobs[job_id]["__delays__"] = []

    def process_job(self, job_id: str, Processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]

        for attempt in range(4):
            try:
                # Call processor with the original data dictionary
                Processor(data)
                return True
            except Exception:
                # If we haven't reached the limit of 3 retries (total 4 attempts)
                if attempt < 3:
                    # Calculate exponential backoff: 2^0=1, 2^1=2, 2^2=4
                    delay = 2 ** attempt
                    data["__retries__"] += 1
                    data["__delays__"].append(delay)
                else:
                    # All 4 attempts failed
                    return False
        return False