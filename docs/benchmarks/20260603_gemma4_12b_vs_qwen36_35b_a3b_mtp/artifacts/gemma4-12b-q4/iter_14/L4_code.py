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

    def process_job(self, job_id, processor):
        # Step 1: Atomically check and remove the job from the queue.
        # By removing it now, we ensure no other thread can process 
        # the same job_id simultaneously.
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # Step 2: Perform the actual processing.
        # We do this outside the lock so that multiple threads can 
        # process different jobs in parallel without blocking each other.
        result = processor(data)

        # Step 3: Atomically store the result.
        with self.lock:
            self.results[job_id] = result
            return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)