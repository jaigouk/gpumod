import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to protect shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock to ensure thread-safe addition to the queue
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Lock to perform the check-and-remove operation atomically.
        # This prevents multiple threads from processing the same job.
        with self.lock:
            if job_id not in self.jobs:
                return False

            # pop() is atomic: it retrieves the value and removes the key simultaneously.
            # This effectively "claims" the job for this thread.
            data = self.jobs.pop(job_id)

        # Perform the actual processing outside the lock.
        # This allows other threads to access the queue (add/get jobs) 
        # while this specific job is being processed.
        result = processor(data)

        # Lock again to safely store the result
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        # Lock to ensure consistent reading of results
        with self.lock:
            return self.results.get(job_id)