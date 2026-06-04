import threading

    class JobQueue:
        def __init__(self):
            self.jobs = {}
            self.results = {}
            self.lock = threading.Lock()

        def add_job(self, job_id, data):
            with self.lock:
                self.jobs[job_id] = data
            return job_id

        def process_job(self, job_id, processor):
            # Step 1: Extract job data and remove it from queue so no one else grabs it
            with self.lock:
                if job_id not in self.jobs:
                    return False
                data = self.jobs.pop(job_id) # Atomic pop inside lock

            # Step 2: Run processing OUTSIDE the lock to allow concurrency
            result = processor(data)

            # Step 3: Store result
            with self.lock:
                self.results[job_id] = result
            return True

        def get_result(self, job_id):
            with self.lock:
                return self.results.get(job_id)