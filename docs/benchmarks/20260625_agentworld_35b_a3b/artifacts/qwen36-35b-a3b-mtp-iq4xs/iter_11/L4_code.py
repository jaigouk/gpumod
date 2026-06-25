import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to protect shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Protect the addition of jobs to the queue
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Lock to safely check for the job and remove it from the queue.
        # This prevents multiple threads from processing the same job.
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs[job_id]
            del self.jobs[job_id]  # Remove job immediately to claim it

        # Process the data outside the lock to allow other threads to access the queue
        # (e.g., add new jobs or process other jobs).
        result = processor(data)

        # Lock to safely write the result
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        # Protect reading the results
        with self.lock:
            return self.results.get(job_id)