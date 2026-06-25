import threading

class JobQueue:
    def __init__(self):
        # Shared data structures that are accessed by multiple threads
        self.jobs = {}      # job_id -> data waiting to be processed
        self.results = {}    # job_id -> result once the job is processed
        self.lock = threading.Lock()   # protects all shared access

    def add_job(self, job_id, data):
        """Put a new job into the queue and return its identifier."""
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        """Process a job and store its result."""
        with self.lock:
            # Ensure the job really exists (pre‑condition)
            if job_id not in self.jobs:
                return False

            # Retrieve the data attached to the job
            data = self.jobs[job_id]

            # Compute the result using the supplied processor
            result = processor(data)

            # Store the result and remove the job from the queue
            self.results[job_id] = result
            del self.jobs[job_id]

        return True

    def get_result(self, job_id):
        """Return the result for a completed job (or None if not yet available)."""
        with self.lock:
            return self.results.get(job_id)