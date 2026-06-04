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

        def process_job(self, job_id, Processor):
            data = None
            with self.lock:
                if job_id not in self.jobs:
                    return False
                # Remove it immediately so no other thread processes it
                data = self.jobs.pop(job_id)

            # Run processor outside the lock to allow concurrency
            result = Processor(data)

            with self.lock:
                self.results[job_id] = result
            return True

        def get_result(self, job_id):
            with self.lock:
                return self.results.get(job_id)