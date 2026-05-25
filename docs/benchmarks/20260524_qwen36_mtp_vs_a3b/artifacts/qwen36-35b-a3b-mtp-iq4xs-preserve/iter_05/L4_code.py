import threading

    class JobQueue:
        def __init__(self):
            self.jobs = {}
            self.results = {}
            # 1. Create a lock
            self.lock = threading.Lock()

        def add_job(self, job_id, data):
            # 2. Acquire lock before modifying shared state
            with self.lock:
                self.jobs[job_id] = data
            return job_id

        def process_job(self, job_id, processor):
            # 3. Acquire lock to check existence and retrieve atomically
            with self.lock:
                if job_id not in self.jobs:
                    return False
                data = self.jobs[job_id]
                # Remove from jobs immediately to prevent re-processing
                del self.jobs[job_id]

            # 4. Process outside the lock (good practice to avoid holding lock during long tasks)
            result = processor(data)

            # 5. Acquire lock to store result
            with self.lock:
                self.results[job_id] = result

            return True

        def get_result(self, job_id):
            # 6. Acquire lock to read
            with self.lock:
                return self.results.get(job_id)