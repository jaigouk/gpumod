import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize the lock to manage thread synchronization
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock the dictionary while adding a job to prevent race conditions
        with self.lock:
            self.jobs[job_id] = data

    def process_job(self, job_id, processor):
        # Lock to atomically check for the job, retrieve data, and remove it from the queue.
        # Removing it here prevents another thread from processing the same job 
        # or adding a new job with the same ID that might overwrite data being processed.
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs[job_id]
            del self.jobs[job_id]

        # Perform the actual processing outside the lock.
        # This allows other threads to continue adding or processing other jobs
        # without waiting for this potentially long-running task to finish.
        result = processor(data)

        # Lock to safely write the result to the results dictionary
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        # Lock to safely read from the results dictionary
        with self.lock:
            return self.results.get(job_id)