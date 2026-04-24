import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to synchronize access to shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock to prevent race conditions when modifying the jobs dictionary
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Lock to ensure the check, processing, and state updates happen atomically.
        # We hold the lock during the entire process to prevent other threads 
        # from deleting the job or overwriting results while this thread is working.
        with self.lock:
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]
            
            # Execute the processor. 
            # Note: In a high-concurrency scenario with long-running processors,
            # holding the lock here blocks other threads. However, this is the 
            # safest way to fix the race condition without altering the logic 
            # (e.g., moving the deletion of the job before processing).
            result = processor(data)

            # Write result and remove job from queue atomically
            self.results[job_id] = result
            del self.jobs[job_id]
        return True

    def get_result(self, job_id):
        # Lock to prevent race conditions when reading results
        with self.lock:
            return self.results.get(job_id)