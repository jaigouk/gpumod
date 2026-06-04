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
            data = None
            with self.lock:
                if job_id not in self.jobs:
                    return False
                data = self.jobs.pop(job_id) # pop ensures only one thread gets the data

            # Process the data outside the lock so other jobs can be processed
            # simultaneously by other threads.
            result = processor(data)

            with self.lock:
                self.results[job_id] = result
            return True

        def get_result(self, job_id):
            with self.lock:
                return self.results.get(job_id)