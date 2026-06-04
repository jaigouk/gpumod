import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to synchronize access to shared dictionaries
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, Processor):
        data = None

        # Critical Section 1: Check and "claim" the job
        with self.lock:
            if job_id not in self.jobs:
                return False
            # pop() removes the item and returns it, 
            # ensuring only one thread can "own" this job_id.
            data = self.jobs.pop(job_id)

        # Perform the actual work OUTSIDE the lock.
        # This allows other threads to process different jobs simultaneously.
        result = Processor(data)

        # Critical Section 2: Save the result
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)