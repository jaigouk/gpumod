import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to manage access to shared resources
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock to ensure adding a job is thread-safe
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Lock to ensure the entire transaction (check, process, store, delete) 
        # is atomic. This prevents multiple threads from processing the same 
        # job or writing results concurrently.
        with self.lock:
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]
            
            # Execute the processor. 
            # Note: Holding the lock during processing blocks other threads 
            # from accessing the queue. For high-concurrency systems, 
            # it is often better to 'pop' the job from the queue before 
            # processing to release the lock, but this preserves the 
            # original logic where the job remains in 'jobs' until the 
            # result is stored (e.g., for retry logic on failure).
            result = processor(data)
            
            self.results[job_id] = result
            del self.jobs[job_id]
        return True

    def get_result(self, job_id):
        # Lock to ensure reading the result is consistent
        with self.lock:
            return self.results.get(job_id)