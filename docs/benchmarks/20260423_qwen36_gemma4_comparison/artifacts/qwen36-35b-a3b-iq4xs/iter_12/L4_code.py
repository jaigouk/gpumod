import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Create a lock to protect shared state
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Lock only around state checks and modifications
        with self._lock:
            if job_id not in self.jobs:
                return False
            # Atomically retrieve and remove the job
            data = self.jobs.pop(job_id)
            
        # Perform the actual work outside the lock to allow concurrent processing
        result = processor(data)
        
        # Lock again to safely store the result
        with self._lock:
            self.results[job_id] = result
            
        return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)