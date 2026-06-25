import threading

class JobQueue:
    def __init__(self):
        # storage for pending jobs and their results
        self.jobs = {}
        self.results = {}
        # lock that protects all mutual access to the two dictionaries
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        """Insert a new job.  The lock guarantees that only one thread writes
        to `self.jobs` at a time, avoiding any overlapping writes."""
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        """Process a pending job and store its result.
        The entire sequence is locked so that reads of `job_id` and writes
        to `self results` cannot interleave in an unsafe way."""
        with self.lock:
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]
            result = processor(data)

            # Store the result and remove the job from the pending set
            self.results[job_id] = result
            del self.jobs[job_id]
        return True

    def get_result(self, job_id):
        """Return the result for a completed job."""
        with self.lock:               # keep the read atomic and exclusive
            return self.results.get(job_id)