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

        # Phase 1: Atomically check and "claim" the job
        with self.lock:
            if job_id not in self.jobs:
                return False

            # pop() removes the item and returns it in one go.
            # This prevents two threads from processing the same ID.
            data = self.jobs.pop(job_id)

        # Phase 2: Run the processor outside the lock.
        # This allows multiple threads to run different jobs in parallel.
        result = processor(data)

        # Phase 3: Atomically save the result
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)