import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to protect shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock to ensure thread-safe addition to the queue
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        with self.lock:
            # Check if job exists
            if job_id not in self.jobs:
                return False
            
            # Extract data and remove from queue immediately to "claim" the job.
            # This prevents other threads from picking up the same job.
            data = self.jobs[job_id]
            del self.jobs[job_id]
        
        # Process the job outside the lock to allow other threads to work concurrently
        result = processor(data)
        
        with self.lock:
            # Lock to ensure thread-safe write to results
            self.results[job_id] = result
            
        return True

    def get_result(self, job_id):
        # Lock to ensure thread-safe read of results
        with self.lock:
            return self.results.get(job_id)