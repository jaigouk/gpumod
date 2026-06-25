import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to protect shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock to ensure thread-safe addition to the jobs dictionary
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        with self.lock:
            # Check if job exists and remove it atomically to prevent 
            # multiple threads from processing the same job.
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]
            del self.jobs[job_id]

        # Process the job outside the lock to allow concurrency.
        # Other threads can now add jobs or process other jobs.
        result = processor(data)

        # Lock again to safely store the result.
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        # Lock to ensure thread-safe reading of results
        with self.lock:
            return self.results.get(job_id)