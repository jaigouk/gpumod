import threading

    class JobQueue:
        def __init__(self):
            self.jobs = {}
            self.results = {}
            # 1. Initialize the lock
            self.lock = threading.Lock()

        def add_job(self, job_id, data):
            # 2. Lock when modifying shared state
            with self.lock:
                self.jobs[job_id] = data
            return job_id

        def process_job(self, job_id, processor):
            # 3. Lock when checking and modifying shared state
            with self.lock:
                if job_id not in self.jobs:
                    return False

                # We need to remove it so it's not processed twice
                # or to signal it's taken.
                data = self.jobs.pop(job_id, None) # pop is atomic and safe

            if data is None:
                return False

            # 4. Process OUTSIDE the lock to allow concurrency
            result = processor(data)

            # 5. Lock to save result
            with self.lock:
                self.results[job_id] = result

            return True

        def get_result(self, job_id):
            with self.lock:
                return self.results.get(job_id)