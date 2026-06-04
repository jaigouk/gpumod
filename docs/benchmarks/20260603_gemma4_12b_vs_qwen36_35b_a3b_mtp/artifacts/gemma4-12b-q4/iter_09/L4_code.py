import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to synchronize access to the dictionaries
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Step 1: Atomically check if the job exists and "claim" it.
        # We use .pop() to remove the job from the 'jobs' dict immediately.
        # This ensures that no other thread can process the same job_id.
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # Step 2: Execute the processor.
        # This is done OUTSIDE the lock so that multiple threads can 
        # execute different jobs in parallel.
        try:
            result = processor(data)
        except Exception as e:
            # Optional: Handle or log processing errors here
            raise e

        # Step 3: Atomically store the result.
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)