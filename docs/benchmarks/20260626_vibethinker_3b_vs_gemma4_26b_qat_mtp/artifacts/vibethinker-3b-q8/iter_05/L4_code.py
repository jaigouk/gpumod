import threading

class JobQueue:
    """
    A simple job queue that can be used in a multi‑threaded environment.
    All mutation operations on the internal dictionaries are protected by
    a locking thread‑safe object, eliminating the race condition.
    """

    def __init__(self):
        self.jobs = {}      # job_id → data
        self.results = {}    # job_id → processed result
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        """
        Put a new job into the queue and return its id so the caller can
        enqueue it on the work queue.
        """
        with self.lock:                 # lock protects the dict update
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        """
        Pick up a job, process it, and store the result.
        Returns True if a job was processed, False if the job id was not found.
        """
        with self.lock:                 # critical section: read, compute, write
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]   # retrieve the data (the dict is locked)
            result = processor(data)    # computation is thread‑safe here
            self.results[job_id] = result
            del self.jobs[job_id]       # remove the job once it's handled
        return True

    def get_result(self, job_id):
        """
        Return the result for a completed job.
        If the result has not yet been written, None is returned.
        """
        with self.lock:                 # read is also protected for consistency
            return self.results.get(job_id, None)