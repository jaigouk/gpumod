import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to manage thread safety
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock to ensure dictionary modification is atomic
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        with self.lock:
            # Check if job exists
            if job_id not in self.jobs:
                return False
            
            # Retrieve data and immediately remove it from the queue.
            # This ensures that only one thread can process this specific job_id.
            data = self.jobs[job_id]
            del self.jobs[job_id]
        
        # Process the data outside the lock.
        # This is important for performance; holding the lock during
        # potentially long-running processing would block other threads.
        try:
            result = processor(data)
        except Exception:
            # Optional: Handle processor errors. 
            # In the original code, an exception here would crash the thread 
            # but leave the job deleted.
            return False

        with self.lock:
            # Lock again to safely write the result
            self.results[job_id] = result
            
        return True

    def get_result(self, job_id):
        # Lock to ensure we read a consistent state
        with self.lock:
            return self.results.get(job_id)