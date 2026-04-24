import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Create a lock object to manage thread synchronization
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Acquire lock before modifying shared state
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Acquire lock to ensure atomic check, read, write, and delete operations
        with self.lock:
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]
            
            # Note: In a high-performance system, you might release the lock 
            # here before calling processor(data) to allow concurrency, 
            # but for this fix, we lock the critical section of state management.
            result = processor(data)

            # The race condition is now fixed because this write is protected by the lock
            self.results[job_id] = result

            del self.jobs[job_id]
        return True

    def get_result(self, job_id):
        # Acquire lock before reading shared state
        with self.lock:
            return self.results.get(job_id)