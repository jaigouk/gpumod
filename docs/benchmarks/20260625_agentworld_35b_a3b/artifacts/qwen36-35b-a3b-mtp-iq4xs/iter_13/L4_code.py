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
        with self.lock:
            if job_id not in self.jobs:
                return False
            # Atomically retrieve and remove the job under the lock
            data = self.jobs.pop(job_id)

        # Execute processor outside the lock to prevent holding it during 
        # potentially slow or blocking operations
        result = processor(data)

        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)