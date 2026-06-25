import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}      # job_id -> data
        self.results = {}    # job_id -> result
        self.lock = threading.Lock()   # protects all concurrent access

    def add_job(self, job_id, data):
        """Add a new job.  Acquire the lock to guarantee unique ID handling."""
        with self.lock:
            # If the same ID is sent again, we simply ignore it (or could raise).
            if job_id not in self.jobs:
                self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        """Process a job and store its result.  All dict access is locked."""
        with self.lock:
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]          # fetch the data
            result = processor(data)          # compute the result
            self.results[job_id] = result      # write the result (atomic with lock)
            del self.jobs[job_id]            # remove the completed job
        return True

    def get_result(self, job_id):
        """Return the result for a job, if it exists."""
        with self.lock:
            return self.results.get(job_id)