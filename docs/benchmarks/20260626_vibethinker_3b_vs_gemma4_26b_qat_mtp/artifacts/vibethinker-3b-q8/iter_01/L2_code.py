from typing import Callable, Dict

class JobQueue:
    def __init__(self):
        # Stores the data for each job and the number of attempts already made
        self._jobs: Dict[str, Dict] = {}
        self._retries: Dict[str, int] = {}

    def add_job(self, job_id: str, data: Dict) -> None:
        """Add a job to be processed."""
        self._jobs[job_id] = data
        self._retries[job_id] = 0

    def process_job(self, job_id: str, processor: Callable)[None] -> bool:
        """
        Process a job with up to 4 attempts, using exponential backoff to
        simulate delays by recording them in the data dictionary.
        Returns True on first successful execution, False if all attempts fail.
        """
        data = self._jobs.get(job_id)
        if data is None:
            raise KeyError(f"Job {job_id} not found")

        attempts = self._retries.get(job_id, 0)

        while attempts < 4:
            # Determine exponential backoff delay for the next attempt (seconds)
            next_attempt = attempts + 1
            if next_attempt > 1:
                # Delays: 1, 2, 4 seconds for attempts 2, 3, 4
                delay = [1, 2, 4][next_attempt - 2]
            else:
                delay = 0

            # Record the delay in the data dict (if non‑zero)
            if 'delays' not in data:
                data['delays'] = []
            if delay:
                data['delays'].append(delay)

            try:
                # Execute the processor; success is defined as no exception raised
                processor(data)
                return True
            except Exception:
                # Failure – increment attempt count and repeat
                attempts += 1
                self._retries[job_id] = attempts
                # No actual sleep; the delay has been recorded in `data`

        # All four attempts exhausted
        self._retries[job_id] = 4
        return False