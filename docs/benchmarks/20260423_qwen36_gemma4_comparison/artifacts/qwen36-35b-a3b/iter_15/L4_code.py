import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Acquire lock to safely read and remove the job
        with self._lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)
            
        # Execute processor outside the lock to avoid blocking other threads
        result = processor(data)
        
        # Acquire lock to safely store the result
        with self._lock:
            self.results[job_id] = result
        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)