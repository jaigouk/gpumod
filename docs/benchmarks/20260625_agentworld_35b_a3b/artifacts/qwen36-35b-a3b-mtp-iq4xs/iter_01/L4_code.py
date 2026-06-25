import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to protect shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock to ensure thread-safe addition to the jobs dictionary
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Lock to atomically check existence, retrieve data, and remove the job
        # This prevents multiple threads from processing the same job simultaneously
        with self.lock:
            if job_id not in self.jobs:
                return False

            # Retrieve data and remove from queue to mark as processing
            data = self.jobs[job_id]
            del self.jobs[job_id]

        # Process the job outside the lock. 
        # This allows other threads to add jobs or process other jobs while this one runs.
        # Note: If the processor raises an exception, the job is removed from the queue
        # but no result is stored.
        result = processor(data)

        # Lock to safely write the result to the results dictionary
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        # Lock to ensure consistent read access to the results dictionary
        with self.lock:
            return self.results.get(job_id)