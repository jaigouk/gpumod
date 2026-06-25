from typing import Callable, List, Dict

class JobQueue:
    def retry(self, job_id: str, processor: Callable) -> bool:
        # Retrieve the job's data from the queue.
        data = self.data_by_id(job_id)
        if data is None:
            raise KeyError(f'Job {job_id!r} not found in the queue.')

        # Prepare slots for retry tracking inside the data dict.
        delays: List[int] = data.setdefault('retry_delays', [])
        data.setdefault('retry_count', 0)

        for attempt in range(4):
            try:
                # Execute the processor with the original data.
                processor(data)
                # Record the number of attempts (1‑based).
                data['retry_count'] = attempt + 1
                return True
            except Exception:
                # If there are further attempts, log the exponential back‑off delay.
                if attempt < 3:
                    delays.append(2 ** attempt)   # 1, 2, 4 seconds

        # All four attempts failed.
        data['retry_count'] = 4
        data['retry_delays'] = [1, 2, 4]
        return False