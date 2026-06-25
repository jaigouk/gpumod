from typing import Callable, Dict

class JobQueue:
    def __init__(self):
        self._jobs: Dict[str, Dict] = {}
        self._retry_counts: Dict[str, int] = {}

    def add_job(self, job_id: str, data: Dict) -> None:
        # Store a shallow copy so modifications to the external data don't affect internal state
        self._jobs[job_id] = data.copy()
        # Initialize retry count for this job
        self._retry_counts[job_id] = 0

    def process_job(self, job_id: str, processor: Callable) -> bool:
        data = self._jobs.pop(job_id, None)
        if data is None:
            return False

        # Prepare a 'delays' key in the data to record exponential backoff values
        data.setdefault('delays', {})

        attempt = 1
        while attempt <= 4:
            try:
                # Execute the processor; any exception will be caught here
                processor(data)
                # Success – record the attempt number and exit
                self._retry_counts[job_id] = attempt
                return True
            except Exception:
                # Record exponential backoff delay for the next attempt (1s, 2s, 4s)
                if attempt < 4:
                    delay = 2 ** (attempt - 1)  # 1, 2, 4
                    data['delays'][delay] = True
                # Update retry count for this job (failure attempt)
                self._retry_counts[job_id] = attempt
                attempt += 1

        # All four attempts failed
        self._retry_counts[job_id] = 4
        return False