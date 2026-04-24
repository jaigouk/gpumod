import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to synchronize access to shared resources
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock to ensure thread-safe addition of jobs
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # We must lock the check-and-remove operation to prevent
        # multiple threads from picking the same job (double processing).
        with self.lock:
            if job_id not in self.jobs:
                return False
            # pop() removes the item from the dictionary atomically.
            # This claims the job for this thread so others won't take it.
            data = self.jobs.pop(job_id)
        
        # Perform the actual processing outside the lock.
        # This allows other threads to pick up and process other jobs
        # while this one is running, maximizing concurrency.
        result = processor(data)

        # Lock again to safely write the result to the shared dictionary
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        # Lock to safely read the result
        with self.lock:
            return self.results.get(job_id)