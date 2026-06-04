import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to synchronize access to shared dictionaries
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, Processor):
        # Use a lock to check for existence and remove the job atomically.
        # By using .pop(), we ensure that only one thread can ever 
        # "claim" a specific job_id.
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # The processor is called OUTSIDE the lock.
        # This is crucial: it allows multiple threads to run 
        # different processors simultaneously.
        try:
            result = Processor(data)
        except Exception as e:
            # Optional: handle errors here (e.g., log them or 
            # put the job back into a failed queue)
            raise e

        # Re-acquire the lock to write the final result.
        with self.lock:
            self.results[job_id] = result
        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)