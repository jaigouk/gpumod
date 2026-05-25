import threading

    class JobQueue:
        def __init__(self):
            self.jobs = {}
            self.results = {}
            self.lock = threading.Lock() # 1. Initialize lock

        def add_job(self, job_id, data):
            # 2. Acquire lock before modifying shared state
            with self.lock:
                self.jobs[job_id] = data
            return job_id

        def process_job(self, job_id, processor):
            # 3. Acquire lock
            with self.lock:
                # Check and delete must be atomic
                if job_id not in self.jobs:
                    return False

                data = self.jobs[job_id]
                del self.jobs[job_id] # Remove immediately to prevent double processing

            # 4. PROCESSING SHOULD HAPPEN OUTSIDE THE LOCK
            # Why? If processor(data) takes 10 seconds, we hold the lock for 10 seconds.
            # This blocks all other threads from adding jobs or getting results.
            # This is a performance bug introduced by naive locking.
            
            result = processor(data)

            with self.lock:
                self.results[job_id] = result
            
            return True

        def get_result(self, job_id):
            with self.lock:
                return self.results.get(job_id)