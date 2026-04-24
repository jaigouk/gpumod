import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Create a lock to protect shared resources
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock to ensure thread-safe addition to the dictionary
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Lock the entire critical section.
        # We must hold the lock while processing because the original logic 
        # keeps the job in self.jobs until the result is written. 
        # If we released the lock during processing, another thread could 
        # pick up the same job_id, leading to double processing and race conditions.
        with self.lock:
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]
            result = processor(data)
            
            # BUG FIX: This write is now protected by the lock
            self.results[job_id] = result
            
            del self.jobs[job_id]
        return True

    def get_result(self, job_id):
        # Lock to ensure we read a consistent state of results
        with self.lock:
            return self.results.get(job_id)