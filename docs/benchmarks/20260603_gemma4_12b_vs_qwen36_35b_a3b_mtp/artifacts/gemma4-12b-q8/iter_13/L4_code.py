import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to synchronize access to the dictionaries
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        data = None

        # Phase 1: Atomic "Claiming" of the job
        with self.lock:
            if job_id not in self.jobs:
                return False

            # Use pop() to remove the job from the pending dictionary 
            # immediately. This ensures that no other thread can 
            # pick up the same job_id simultaneously.
            data = self.jobs.pop(job_id)

        # Phase 2: Processing
        # We perform the actual processing OUTSIDE the lock.
        # This allows multiple threads to run different processors 
        # concurrently.
        result = processor(data)

        # Phase 3: Atomic storage of the result
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)