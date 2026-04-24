import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to synchronize access to shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Protect adding jobs to the queue
        with self.lock:
            self.jobs[job_id] = data
            return job_id

    def process_job(self, job_id, processor):
        # Protect the entire processing sequence.
        # This ensures that no other thread can interfere with the job's lifecycle
        # (checking existence, processing, writing result, deleting job).
        with self.lock:
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]
            
            # Note: In a high-performance production system, you might want to 
            # release the lock while the processor runs to allow other threads 
            # to work. However, to strictly fix the race condition in this 
            # specific snippet while preserving the original logic (deleting the 
            # job only after processing), we keep the lock held. 
            # Alternatively, one could 'pop' the job inside the lock, process 
            # outside, and then write the result inside a lock, but that changes 
            # the error-handling behavior (job loss on failure).
            
            result = processor(data)

            # Fixed: Protected write to results to prevent race condition
            self.results[job_id] = result

            del self.jobs[job_id]
            return True

    def get_result(self, job_id):
        # Protect reading results
        with self.lock:
            return self.results.get(job_id)