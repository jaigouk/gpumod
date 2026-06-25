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
            # Atomic check-and-remove prevents the TOCTOU race condition
            data = self.jobs.pop(job_id, None)
            if data is None:
                return False

        # Execute processor OUTSIDE the lock to avoid blocking other threads
        result = processor(data)

        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)