from typing import Callable

class JobQueue:
    def retry(self, job_id: str, processor: Callable) -> bool:
        data = self.data.get(job_id)
        if data is None:
            return False
        failures = self._retry_counts.get(job_id, 0)
        for attempt in range(1, 5):
            try:
                processor(data)
                return True
            except Exception:
                failures += 1
                if failures >= 4:
                    self._retry_counts[job_id] = failures
                    return False
                delay = 2 ** (failures - 1)
                data.setdefault('delay', []).append(delay)
                self._retry_counts[job_id] = failures
        self._retry_counts[job_id] = failures
        return False