import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to synchronize access to the internal dictionaries
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        data = None

        # Step 1: Atomically check and "claim" the job
        with self.lock:
            if job_id not in self.jobs:
                return False

            # .pop() removes the item and returns it in one step.
            # This ensures that only one thread can successfully 
            # claim a specific job_id.
            data = self.jobs.pop(job_id)

        # Step 2: Process the job OUTSIDE the lock.
        # This allows multiple threads to run the 'processor' 
        # function concurrently for different jobs.
        result = processor(data)

        # Step 3: Atomically store the result
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)