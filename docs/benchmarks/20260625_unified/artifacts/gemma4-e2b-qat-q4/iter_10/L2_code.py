from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        # job_id -> {'data': Any, 'attempts': int}
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]):
        """Adds a job to the queue."""
        self._jobs[job_id] = {
            "data": data,
            "attempts": 0
        }

    def process_job(self, job_id: str, processor: Callable[[Dict], Any]) -> bool:
        """
        Processes a job with up to 4 attempts and exponential backoff delays.
        """
        if job_id not in self._jobs:
            return False

        job_info = self._jobs[job_id]
        job_data = job_info["data"]

        for attempt in range(4):
            try:
                # 1. Call processor(data)
                processor(job_data)

                # 5. Return True on first successful attempt
                return True

            except Exception as e:
                # 2. On any exception from the processor, retry.

                if attempt < 3:
                    # 3. Exponential backoff calculation (1s, 2s, 4s)
                    # Attempt 0: delay 1 (2^0)
                    # Attempt 1: delay 2 (2^1)
                    # Attempt 2: delay 4 (2^2)
                    delay = 2 ** attempt

                    # 4. Do NOT actually sleep — record the delays as data.
                    # In a production environment, this logic would trigger a sleep/wait here.
                    pass

                # If this was the last attempt (attempt 3), we break out of the loop
                if attempt == 3:
                    break

        # 5. Return False if all 4 attempts raise
        return False