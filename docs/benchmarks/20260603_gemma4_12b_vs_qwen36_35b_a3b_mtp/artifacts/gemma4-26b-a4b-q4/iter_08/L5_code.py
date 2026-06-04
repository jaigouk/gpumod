import time
    import heapq
    from dataclasses import dataclass, field
    from typing import Any, Callable

    @dataclass(order=True)
    class Job:
        priority: int
        data: Any = field(compare=False)

    class PriorityQueue:
        def __init__(self):
            self._items = []
        def push(self, job: Job):
            heapq.heappush(self._items, job)
        def pop(self) -> Job:
            return heapq.heappop(self._items)
        def empty(self):
            return len(self._items) == 0

    class JobQueue:
        def __init__(self, queue_backend=None):
            self.queue = queue_backend or PriorityQueue()
        def add(self, job: Job):
            self.queue.push(job)
        def get(self) -> Job:
            return self.queue.pop()

    def process_with_retry(job_func: Callable, job: Job, max_retries: int = 3):
        retries = 0
        while retries < max_retries:
            try:
                return job_func(job.data)
            except Exception as e:
                retries += 1
                if retries == max_retries:
                    raise e
                wait = 2 ** retries
                print(f"Retrying in {wait}s...")
                time.sleep(wait)