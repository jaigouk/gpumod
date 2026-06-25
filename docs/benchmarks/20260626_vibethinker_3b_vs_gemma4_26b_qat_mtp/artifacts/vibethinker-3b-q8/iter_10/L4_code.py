import threading

class JobQueue:
    """
    A thread‑safe job queue that stores pending jobs in a dict,
    processes them, and records results.  All mutual exclusive
    access to the internal dictionaries is guarded by a single
    threading.Lock to avoid the race condition.
    """

    def __init__(self):
        self.jobs = {}      # job_id -> data to be processed
        self.results = {}    # job_id -> result of processing
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        """
        Put a new job onto the queue.
        Returns the job identifier so the caller can look up the job.
        """
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        """
        Process a job with the given processor.
        Returns True if the job was processed, False if it wasn't found.
        """
        with self._lock:
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]
            result = processor(data)

            # Store the result and remove the job from the queue.
            self.results[job_id] = result
            del self.jobs[job_id]
        return True

    def get_result(self, job_id):
        """
        Return the result for a completed job, or None if not yet computed.
        """
        with self._lock:
            return self.results.get(job_id)