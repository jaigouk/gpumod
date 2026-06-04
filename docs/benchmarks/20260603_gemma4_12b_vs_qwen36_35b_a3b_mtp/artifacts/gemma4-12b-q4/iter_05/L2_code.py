from typing import Callable, Dict, Any

.
    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Any] = {}
            self.retry_counts: Dict[str, int] = {}

        def add_job(self, job_id: str, data: Any):
            self.jobs[job_id] = data
            self.retry_counts[job_id] = 0

        def process_job(self, job_id: str, processor: Callable) -> bool:
            data = self.jobs.get(job_id)
            if data is None:
                return False

            max_retries = 3
            for attempt in range(max_retries + 1):
                try:
                    processor(data)
                    self.retry_counts[job_id] = 0 # Reset or keep? Usually reset on success.
                    return True
                except Exception as e:
                    if attempt < max_retries:
                        self.retry_counts[job_id] += 1
                        # Backoff: 1, 2, 4...
                        delay = 2 ** attempt # 0->1, 1->2, 2->4
                        # Instruction says "tracked rather than actually sleeping"
                        # So we don't call time.sleep()
                        pass
                    else:
                        return False
            return False