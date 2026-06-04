import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to synchronize access to the shared dictionaries
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, Processor):
        # 1. Atomically check if job exists and "claim" it by removing it 
        # from the active jobs dictionary.
        with self.lock:
            if job_id not in self.jobs:
                return False
            # .pop() removes the item and returns it in one atomic step 
            # inside the lock, preventing other threads from grabbing it.
            data = self.jobs.pop(job_id)

        # 2. Perform the actual work OUTSIDE the lock. 
        # This allows multiple threads to run different processors 
        # simultaneously without blocking each other.
        result = Processor(data)

        # 3. Atomically write the result.
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)