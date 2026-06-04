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
            # Step 1: Safely extract data from jobs
            with self.lock:
                if job_id not in self.jobs:
                    return False
                data = self.jobs.pop(job_id) # pop is atomic-ish but we are inside lock

            # Step 2: Execute processor (Outside lock to allow concurrency)
            result = processor(data)

            # Step 3: Store result
            with self.lock:
                self.results[job_id] = result
            return True

        def get_result(self, job_id):
            with self.lock:
                return self.results.get(job_id)