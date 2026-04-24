import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to manage thread access to shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock ensures that adding a job is thread-safe
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Lock ensures that the check, processing, and cleanup are atomic.
        # This prevents multiple threads from processing the same job
        # and prevents race conditions when writing to self.results.
        with self.lock:
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]
            
            # Execute the processor
            # Note: The lock is held during processing. 
            # For high-performance scenarios with long-running tasks, 
            # one might 'pop' the job here to release the lock, 
            # but that changes the semantics (job is lost if processor fails).
            result = processor(data)

            # Update results and remove the job from the active queue
            self.results[job_id] = result
            del self.jobs[job_id]
            
        return True

    def get_result(self, job_id):
        # Lock ensures consistent reading of results
        with self.lock:
            return self.results.get(job_id)