from dataclasses import dataclass, field
        import time
        import heapq
        from typing import Any, Callable, Optional

        @dataclass(order=True)
        class Job:
            priority: int
            data: Any = field(compare=False)

        class JobQueue:
            def __init__(self):
                self.jobs = []
            def push(self, job: Job):
                self.jobs.append(job)
            def pop(self) -> Job:
                return self.jobs.pop(0)
            def is_empty(self):
                return len(self.jobs) == 0

        class PriorityQueue(JobQueue):
            def push(self, job: Job):
                heapq.heappush(self.jobs, job)
            def pop(self) -> Job:
                return heapq.heappop(self.jobs)

        def process_with_retry(job: Job, func: Callable, max_retries: int = 3):
            attempt = 0
            while attempt < max_retries:
                try:
                    return func(job.data)
                except Exception as e:
                    attempt += 1
                    if attempt == max_retries:
                        raise e
                    time.sleep(2 ** attempt)