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
            # We need to ensure that only one thread processes a specific job_id
            # and that dictionary access is thread-safe.
            with self.lock:
                if job_id not in self.jobs:
                    return False
                # We "pop" it here to ensure no other thread picks it up
                data = self.jobs.pop(job_id)

            result = Processor(data)

            with self.lock:
                self.results[job_id] = result
            return True

        def get_result(self, job_id):
            with self.lock:
                return self.results.get(job_id)