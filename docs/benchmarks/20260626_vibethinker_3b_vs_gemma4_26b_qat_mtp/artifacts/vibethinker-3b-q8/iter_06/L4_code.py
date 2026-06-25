import threading

class JobQueue:
    """
    A simple thread‑safe job queue.
    - `add_job`   : enqueue a new job.
    - `process_job`: dequeue a job, execute it and store the result.
    - `get_result`: retrieve a completed result.
    All accesses to the internal dictionaries are protected by a lock
    to eliminate the race condition.
    """

    def __init__(self):
        self.jobs      = {}   # job_id → data to be processed
        self.results    = {}   # job_id → result of the processed job
        self.lock = threading.Lock()

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def add_job(self, job_id, data):
        """Put a new job into the queue and return its id (for look‑up)."""
        with self.lock:               # protect the dictionary
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        """Take a job with *job_id* out of the queue, process it and store the result."""
        with self.lock:               # critical section – read, compute, write
            if job_id not in self.jobs:
                # The job may have been already processed or never enqueued.
                return False

            data = self.jobs[job_id]     # read the data
            result = processor(data)      # compute the result

            # Write the result and clear the queue entry.
            self.results[job_id] = result
            del self.jobs[job_id]
        return True

    def get_result(self, job_id):
        """Return the result for *job_id* or None if it hasn't been produced yet."""
        with self.lock:               # protect the dictionary when reading
            return self.results.get(job_id)

    # -----------------------------------------------------------------
    # Helper – nice string representation (optional)
    # -----------------------------------------------------------------
    def __repr__(self):
        return f"JobQueue(jobs={self.jobs}, results={self.results})"