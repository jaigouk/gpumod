import time
import threading
from typing import Any, Callable, Optional

class Job:
    def __init__(self, job_id: str, func: Callable, args=(), kwargs=None):
        self.job_id = job_id
        self.func = func
        self.args = args
        self.kwargs = kwargs or {}
        self.result = None
        self.error = None
        self.status = "pending"
        self.attempts = 0
        self._lock = threading.Lock()
        self._completed = threading.Event()

    def execute(self):
        return self.func(*self.args, **self.kwargs)

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.lock = threading.Lock()

    def enqueue(self, job_id: str, func: Callable, *args, **kwargs):
        with self.lock:
            self.jobs[job_id] = Job(job_id, func, args, kwargs)

    def process(self, job_id: str):
        with self.lock:
            if job_id not in self.jobs:
                return
            job = self.jobs[job_id]
            job.status = "running"

        max_retries = 3
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                result = job.execute()
                with job._lock:
                    job.result = result
                    job.status = "success"
                    job._completed.set()
                return
            except Exception as e:
                last_error = e
                with job._lock:
                    job.attempts = attempt + 1
                    job.error = e

                if attempt < max_retries:
                    backoff = 2 ** attempt
                    time.sleep(backoff)
                else:
                    with job._lock:
                        job.status = "failed"
                        job._completed.set()

    def get_result(self, job_id: str) -> Optional[Any]:
        with self.lock:
            if job_id not in self.jobs:
                raise ValueError("Job not found")
            job = self.jobs[job_id]

        job._completed.wait()

        with job._lock:
            if job.status == "failed":
                raise job.error
            return job.result
