import threading

    class JobQueue:
        def __init__(self):
            self.jobs = {}
            self.results = {}
            self.lock = threading.Lock() # Added lock

        def add_job(self, job_id, data):
            with self.lock: # Thread-safe write
                self.jobs[job_id] = data
            return job_id

        def process_job(self, job_id, processor):
            with self.lock:
                if job_id not in self.jobs:
                    return False
                data = self.jobs[job_id]
                # Should we keep the lock during processor(data)?
                # If we do, we block other jobs.
                # If we don't, two threads could still run the same job.
                # To prevent two threads from running the same job,
                # we should remove it from self.jobs immediately.
                del self.jobs[job_id]

            # Run processor outside the lock to allow concurrency
            result = processor(data)

            with self.lock:
                self.results[job_id] = result
            return True

        def get_result(self, job_id):
            with self.lock: # Safe read
                return self.results.get(job_id)