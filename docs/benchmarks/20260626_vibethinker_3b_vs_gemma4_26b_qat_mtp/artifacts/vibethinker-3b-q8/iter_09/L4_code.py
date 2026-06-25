import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}      # job_id -> data
        self.results = {}    # job_id -> result
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        """Enqueue a new job and return its identifier."""
        with self.lock:               # protect the dictionary
            self.jobs[job_id] = data
            return job_id

    def process_job(self, job_id, processor):
        """Process a pending job and store its result."""
        with self.lock:
            if job_id not in self.jobs:
                return False          # job already processed or not present

            # Remove the job from the queue *before* computing the result
            data = self.jobs.pop(job_id)

            result = processor(data)  # processor may block, but it does not break the lock
            self.results[job_id] = result   # write the result atomically
        return True

    def get_result(self, job_id):
        """Return the result for a completed job (or None)."""
        with self.lock:               # optional – makes reading thread‑safe as well
            return self.results.get(job_id)