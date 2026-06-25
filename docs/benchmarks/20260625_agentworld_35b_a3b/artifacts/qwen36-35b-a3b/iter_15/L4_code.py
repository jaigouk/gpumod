import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to protect shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock to ensure safe addition to the jobs dictionary
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. Lock to atomically check and remove the job from the queue.
        # This prevents multiple threads from processing the same job.
        with self.lock:
            if job_id not in self.jobs:
                return False
            # pop() removes the item and returns its value atomically.
            # This effectively "claims" the job so no other thread can pick it up.
            data = self.jobs.pop(job_id)

        # 2. Process the job outside the lock.
        # This allows other threads to add jobs or process other jobs concurrently.
        # Note: If the processor raises an exception, the job is lost 
        # because it was removed from self.jobs. This is a standard trade-off 
        # for concurrency in queues (or requires explicit error handling/retry logic).
        result = processor(data)

        # 3. Lock to safely write the result to the results dictionary.
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        # Lock to ensure safe reading of results
        with self.lock:
            return self.results.get(job_id)