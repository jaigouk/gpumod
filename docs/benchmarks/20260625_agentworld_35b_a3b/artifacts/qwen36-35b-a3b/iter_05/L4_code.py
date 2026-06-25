import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Lock around check-and-fetch to prevent TOCTOU race conditions
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # Process outside the lock to allow concurrent job execution
        result = processor(data)

        # Lock around result storage to prevent concurrent writes
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)