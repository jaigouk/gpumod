import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize the lock to manage thread safety
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock to ensure safe insertion into the jobs dictionary
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Lock to safely check existence and retrieve the job atomically
        with self.lock:
            if job_id not in self.jobs:
                return False
            # pop() removes the item and returns it, preventing race conditions
            # where another thread might delete the job between check and access
            data = self.jobs.pop(job_id)

        # Perform processing outside the lock to allow concurrency
        # (Other threads can add jobs or check results while this is running)
        result = processor(data)

        # Lock to safely store the result
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        # Lock to ensure consistent read of the results dictionary
        with self.lock:
            return self.results.get(job_id)