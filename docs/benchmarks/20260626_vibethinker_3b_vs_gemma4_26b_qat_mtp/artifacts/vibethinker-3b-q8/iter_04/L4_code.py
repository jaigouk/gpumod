import threading

class JobQueue:
    """A simple thread‑safe job queue."""

    def __init__(self):
        self.jobs = {}      # job_id -> data waiting to be processed
        self.results = {}    # job_id -> result of processing
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        """Add a new job to the queue.

        Returns a unique identifier so the caller can look up the job.
        """
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        """Process a pending job and remember its result.

        Returns True if the job existed and was successfully processed,
        otherwise False.
        """
        with self.lock:
            # Check whether the job is still present.
            if job_id not in self.jobs:
                return False
            # Retrieve the data and run the processor.
            data = self.jobs[job_id]
            result = processor(data)
            # Remove the job from the pending set once it has been processed.
            del self.jobs[job_id]

        # Now that the job is cleared from `jobs`, we can safely store the result.
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        """Return the result for a job that has already been processed.

        If the job never existed or wasn't processed, this will return None.
        """
        return self.results.get(job_id)