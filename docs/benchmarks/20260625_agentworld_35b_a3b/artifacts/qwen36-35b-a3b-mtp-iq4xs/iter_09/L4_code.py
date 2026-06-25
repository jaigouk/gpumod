import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to protect shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock to safely add to the shared dictionary
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        with self.lock:
            # Check if job exists and remove it atomically.
            # Removing it here (pop) prevents multiple threads from 
            # processing the same job simultaneously.
            if job_id not in self.jobs:
                return False

            # pop() removes the key and returns the value in one atomic step
            data = self.jobs.pop(job_id)

        # Perform the actual processing outside the lock.
        # This allows other threads to access the queue while this job is being processed.
        result = processor(data)

        with self.lock:
            # Lock to safely write the result to the shared dictionary
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        # Lock to safely read from the shared dictionary
        with self.lock:
            return self.results.get(job_id)