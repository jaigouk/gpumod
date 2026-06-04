import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to synchronize access to shared resources
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Step 1: Atomically check if job exists and "claim" it by removing it
        # We use pop() so that only one thread can successfully retrieve the data
        data = None
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # Step 2: Process the job outside the lock.
        # This allows multiple threads to run different processors in parallel.
        result = processor(data)

        # Step 3: Atomically store the result
        with self.lock:
            self.results[job_id] = result
        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)