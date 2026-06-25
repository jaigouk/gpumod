import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to protect shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock access to the jobs dictionary to prevent concurrent modification issues
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Lock to atomically check for the job and remove it from the queue.
        # Using pop() ensures the job is removed immediately, preventing 
        # another thread from picking it up while we are processing.
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # Execute the processor OUTSIDE the lock.
        # This allows other threads to add jobs or retrieve results 
        # while this job is being processed.
        result = processor(data)

        # Lock to safely write the result to the results dictionary
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        # Lock to safely read from the results dictionary
        with self.lock:
            return self.results.get(job_id)