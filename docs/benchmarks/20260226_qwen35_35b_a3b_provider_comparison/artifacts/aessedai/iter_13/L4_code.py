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
        # Atomically claim the job to prevent concurrent processing
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)  # Remove immediately to prevent reprocessing

        # Process the job outside the lock to allow concurrency
        result = processor(data)

        # Store the result under lock to ensure thread-safe write
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)