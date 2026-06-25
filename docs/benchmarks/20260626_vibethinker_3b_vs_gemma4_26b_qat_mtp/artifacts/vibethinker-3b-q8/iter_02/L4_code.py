import threading

class JobQueue:
    def __init__(self):
        """Initialize an empty job queue."""
        self.jobs = {}   # Map job_id -> data to be processed
        self.results = {} # Map job_id -> result of processed data
        self.lock = threading.Lock()   # Shared lock for all dict accesses

    def add_job(self, job_id, data):
        """
        Enqueue a new job.
        Returns the job_id so the caller can pass it to a processor.
        """
        with self.lock:                # Acquire the lock before mutating the dict
            self.jobs[job_id] = data   # Write the job
        return job_id

    def process_job(self, job_id, processor):
        """
        Process a pending job and store its result.
        Returns True if the job existed and was processed, False otherwise.
        """
        with self.lock:                # Lock ensures safe read‑and‑write sequence
            if job_id not in self_jobs:
                return False

            data = self.jobs[job_id]   # Read the data (safe under lock)
            result = processor(data)   # Processor runs in isolation

            self.results[job_id] = result  # Write the result
            del self.jobs[job_id]       # Remove the job once processed
        return True

    def get_result(self, job_id):
        """
        Retrieve the result for a previously processed job.
        Returns the result or None if it hasn't been produced yet.
        """
        with self.lock:                # lock protects the get / get‑None call
            return self.results.get(job_id)