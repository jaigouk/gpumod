import threading

class JobQueue:
    """
    A simple job queue that safely adds jobs, processes them and fetches results.
    All inter‑thread communication is protected by an instance‑level lock
    to eliminate the race condition.
    """

    def __init__(self):
        self.jobs = {}      # job_id → data to be processed
        self.results = {}    # job_id → result of the processed job
        self._lock = threading.Lock()   # exclusive lock for all mutable accesses

    def add_job(self, job_id, data):
        """
        Put a new job into the queue.
        Returns the job_id so the caller can pass it to a processor.
        """
        with self._lock:                 # lock ensures only one thread touches the dicts at a time
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        """
        Pick the specified job, run the supplied processor on its data
        and store the result.  Returns True iff a job was processed.
        """
        with self._lock:
            if job_id not in self.jobs:
                # The job may have been processed already or never added.
                return False

            data = self.jobs[job_id]    # fetch the data (safe under lock)
            result = processor(data)     # compute the result
            self.results[job_id] = result # store the result
            del self.jobs[job_id]        # remove the job from the queue
        return True

    def get_result(self, job_id):
        """
        Return the result for a completed job, or None if not available.
        """
        with self._lock:
            return self.results.get(job_id)