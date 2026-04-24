import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize the lock to manage thread access
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock to ensure thread-safe addition to the jobs dictionary
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. Lock to safely check existence and remove the job.
        # Using pop() atomically removes the job from the queue, 
        # ensuring no other thread processes it while we hold the data.
        with self.lock:
            if job_id not in self.jobs:
                return False
            # pop() removes the item and returns it
            data = self.jobs.pop(job_id)
        
        # 2. Process the job outside the lock.
        # This allows other threads to add or process other jobs concurrently.
        # Note: If the processor raises an exception, the job is removed 
        # from the queue (consumed) but not saved to results.
        result = processor(data)

        # 3. Lock to safely store the result.
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        # Lock to ensure thread-safe reading from the results dictionary
        with self.lock:
            return self.results.get(job_id)