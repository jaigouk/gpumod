import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Add a lock to synchronize access to shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        with self.lock:
            if job_id not in self.jobs:
                return False
            # Atomically retrieve and remove the job to fix the TOCTOU race condition
            data = self.jobs.pop(job_id)

        # Execute the processor outside the lock to prevent blocking other threads
        # during potentially long-running or I/O-bound tasks
        result = processor(data)

        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)