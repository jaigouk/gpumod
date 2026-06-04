from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.retry_counts: Dict[str, int] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        self.jobs[job_id] = data
        self.retry_counts[job_id] = 0

    def process_job(self, job_id: str, Processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        data = self.jobs[job_id]

        for attempt in range(4):
            try:
                # Requirement 1: Call processor(data)
                Processor(data)
                # Requirement 5: Return True on the first successful attempt
                return True
            except Exception:
                # Requirement 2: On any exception from the processor, retry
                # Requirement 3: Track retry count per job
                # Requirement 4: Record delays as data (do not sleep)
                if attempt < 3:
                    self.retry_counts[job_id] += 1
                    # Exponential backoff (1s, 2s, 4s)
                    delay = 2 ** (self.retry_counts[job_id] - 1)

                    if "delays" not in data:
                        data["delays"] = []
                    data["delays"] .append(delay)
                else:
                    # Requirement 5: Return False if all 4 attempts raise
                    return False
        return False