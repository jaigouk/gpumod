import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to synchronize access to the dictionaries
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, Processor):
        # Phase 1: Atomically claim the job
        with self.lock:
            if job_id not in self.jobs:
                return False

            # We use .pop() to remove the job from the 'jobs' dict 
            # immediately. This ensures that no other thread can 
            # "claim" this same job_id simultaneously.
            data = self.jobs.pop(job_id)

        # Phase 2: Process the data
        # We do this OUTSIDE the lock so that multiple threads can 
        # process different jobs in parallel.
        result = Processor(data)

        # Phase 3: Store the result
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# Example Usage:
if __name____or__self.lock = threading.Lock()