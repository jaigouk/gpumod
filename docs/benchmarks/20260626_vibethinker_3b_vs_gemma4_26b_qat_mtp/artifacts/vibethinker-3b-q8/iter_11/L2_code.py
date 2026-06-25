from typing import Callable, Any
import threading

class JobQueue:
    def __init__(self):
        self._data = {}
        self._retry_counts = {}

    def add_job(self, job_id: str, data: dict):
        self._data[job_id] = data

    def process_job(self, job_id: str, processor: Callable[[dict], Any]) -> bool:
        data = self._data.get(job_id)
        if not data:
            return False

        # If all attempts have been used for this job, give up.
        if self._retry_counts.get(job_id, 0) >= 4:
            self._retry_counts[job_id] = 4
            return False

        success = False
        attempts_done = 0

        def attempt(attempt_num: int) -> None:
            nonlocal success, attempts_done
            attempts_done = attempt_num
            try:
                processor(data)
                if not success:
                    success = True
            except:
                if attempt_num < 4:
                    delay = 2 ** (attempt_num - 1)  # 1, 2, 4 seconds
                    data['next_delay'] = delay
                    threading.Timer(delay, attempt(attempt_num + 1))

        start_attempt = self._retry_counts.get(job_id, 0) + 1
        attempt(start_attempt)

        self._retry_counts[job_id] = attempts_done if success else 4
        return success