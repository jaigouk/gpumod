import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to protect shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Acquire lock to safely modify the jobs dictionary
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Acquire lock to ensure atomic check, process, and delete operations
        with self.lock:
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]
            
            # Note: In a high-performance system, you might release the lock 
            # here before calling processor(data) to avoid blocking other threads,
            # but that would require moving the deletion logic earlier to prevent 
            # the job from being picked up by another thread while processing.
            # For this fix, we keep the logic flow identical to the original 
            # while ensuring thread safety.
            result = processor(data)
            
            # Safely write the result
            self.results[job_id] = result
            
            # Safely remove the job
            del self.jobs[job_id]
        
        return True

    def get_result(self, job_id):
        # Acquire lock to safely read from the results dictionary
        with self.lock:
            return self.results.get(job_id)