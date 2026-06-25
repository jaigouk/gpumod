import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to protect shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock access to self.jobs to prevent race conditions during insertion
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Lock to safely check existence and remove the job (atomic check-and-act)
        with self.lock:
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]
            del self.jobs[job_id]

        # Process the job outside the lock. 
        # This allows other threads to add or process different jobs concurrently.
        # Note: The job has already been removed from self.jobs, so it won't 
        # be picked up by another thread.
        try:
            result = processor(data)
        except Exception as e:
            # Optional: Handle errors. In this implementation, the job is 
            # considered consumed/lost if processing fails, as it was removed 
            # from the queue before processing.
            raise e

        # Lock to safely write the result
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        # Lock to safely read from self.results
        with self.lock:
            return self.results.get(job_id)