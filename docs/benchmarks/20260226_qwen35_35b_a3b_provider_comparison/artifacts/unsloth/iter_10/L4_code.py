import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        """Add a new job to the queue."""
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        """Process a job if it exists in the queue."""
        with self.lock:
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]
            result = processor(data)

            self.results[job_id] = result
            del self.jobs[job_id]
        return True

    def get_result(self, job_id):
        """Retrieve the result of a job."""
        with self.lock:
            return self.results.get(job_id)