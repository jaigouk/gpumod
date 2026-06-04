import time
        import random
        from dataclasses import dataclass, field
        from typing import Any, List
        import heapq

        @dataclass
        class Job:
            id: str
            data: Any
            priority: int = 0

        class JobQueue:
            def __init__(self):
                self.jobs = []
            def add(self, job: Job):
                self.jobs.append(job)
            def get_next(self) -> Job:
                return self.jobs.pop(0) if self.jobs else None

        class PriorityQueue(JobQueue):
            def __init__(self):
                self.jobs = []
            def add(self, job: Job):
                heapq.heappush(self.jobs, (job.priority, job.id, job))
            def get_next(self) -> Job:
                return heapq.heappop(self.jobs)[2] if self.jobs else None

        def process_with_retry(job: Job, max_retries=3):
            attempt = 0
            while attempt < max_retries:
                try:
                    print(f"Processing {job.id}")
                    # Simulate work
                    if random.random() < 0.5: raise Exception("Failed")
                    return True
                except Exception as e:
                    attempt += 1
                    wait = 2 ** attempt
                    print(f"Retry {attempt} in {wait}s")
                    time.sleep(wait)
            return False