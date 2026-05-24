import threading

    class JobQueue:
        def __init__(self):
            self.jobs = {}
            self.results = {}
            self.lock = threading.Lock() # Add lock

        def add_job(self, job_id, data):
            with self.lock: # Protect write
                self.jobs[job_id] = data
            return job_id

        def process_job(self, job_id, processor):
            # We need to claim the job atomically
            with self.lock:
                if job_id not in self.jobs:
                    return False
                data = self.jobs[job_id]
                # Remove immediately so other threads don't pick it up
                del self.jobs[job_id]

            # Process outside the lock to allow concurrency
            result = processor(data)

            # Store result safely
            with self.lock:
                self.results[job_id] = result

            return True

        def get_result(self, job_id):
            with self.lock: # Protect read (optional for dict get, but safe)
                return self.results.get(job_id)