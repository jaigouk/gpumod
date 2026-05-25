import threading

    class JobQueue:
        def __init__(self):
            self.jobs = {}
            self.results = {}
            self.lock = threading.Lock() # Create lock

        def add_job(self, job_id, data):
            with self.lock:
                self.jobs[job_id] = data
            return job_id

        def process_job(self, job_id, processor):
            # Critical section: Check and remove job
            with self.lock:
                if job_id not in self.jobs:
                    return False
                # Using pop is atomic and cleaner
                data = self.jobs.pop(job_id)

            # Do work outside lock
            result = processor(data)

            # Critical section: Store result
            with self.lock:
                self.results[job_id] = result

            return True

        def get_result(self, job_id):
            with self.lock:
                return self.results.get(job_id)