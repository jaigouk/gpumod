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
            # We want to ensure that only one thread can "claim" a job.
            # To allow parallel processing, we should only lock while
            # accessing the shared dictionaries.
            data = None
            with self.lock:
                if job_id not in self.jobs:
                    return False
                # Remove the job immediately so no other thread can pick it up
                data = self.jobs.pop(job_id)

            # Perform the heavy lifting outside the lock
            result = Processor(data)

            with self.lock:
                self.results[job_id] = result
            return True

        def get_result(self, job_id):
            with self.lock:
                return self.results.get(job_id)