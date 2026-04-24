import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Create a lock to protect shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock to safely write to the jobs dictionary
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Lock to safely check existence and retrieve data
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs[job_id]
            # Remove job immediately to prevent other threads from processing it
            del self.jobs[job_id]
        
        # Process data outside the lock to avoid blocking other threads
        try:
            result = processor(data)
        except Exception:
            # Handle potential errors in the processor if necessary
            return False

        # Lock to safely write the result
        with self.lock:
            self.results[job_id] = result
        
        return True

    def get_result(self, job_id):
        # Lock to safely read from the results dictionary
        with self.lock:
            return self.results.get(job_id)