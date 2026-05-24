import threading

    class JobQueue:
        def __init__(self):
            self.jobs = {}
            self.results = {}
            self.lock = threading.Lock() # 1. Create lock

        def add_job(self, job_id, data):
            with self.lock: # 2. Lock access
                self.jobs[job_id] = data

        def process_job(self, job_id, processor):
            with self.lock: # 3. Lock access
                if job_id not in self.jobs:
                    return False

                data = self.jobs[job_id]
                # Note: Processing happens inside the lock.
                # Warning: If processor is slow, this blocks other threads.
                # However, for a simple fix of the race condition described,
                # this is the standard approach.
                result = processor(data)
                self.results[job_id] = result
                del self.jobs[job_id]
            return True

        def get_result(self, job_id):
            with self.lock: # 4. Lock access
                return self.results.get(job_id)