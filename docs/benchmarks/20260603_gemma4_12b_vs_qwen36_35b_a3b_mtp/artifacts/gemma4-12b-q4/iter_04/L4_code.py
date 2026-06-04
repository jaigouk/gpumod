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
        # Step 1: Atomically check existence and remove the job from the queue.
        # Removing it here ensures that no other thread can pick up the same job_id.
        with self.lock:
            if job_id not in self.jobs:
                return False

            # Use pop to retrieve and remove the job in one thread-safe operation
            data = self.jobs.pop(job_id)

        # Step 2: Perform the processing outside of the lock.
        # We release the lock here so that other threads can add/process 
        # other jobs while this specific processor is running.
        result = Processor(data)

        # Step 3: Atomically write the result.
        with self.lock:
            self.results[job_id] = result
            return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)