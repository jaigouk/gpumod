import threading

    class JobQueue:
        def __init__(self):
            self.jobs = {}
            self.results = {}
            self.lock = threading.Lock() # Add lock

        def add_job(self, job_id, data):
            with self.lock:
                self.jobs[job_id] = data
            return job_id

        def process_job(self, job_id, processor):
            # Lock needed here to check existence and remove
            with self.lock:
                if job_id not in self.jobs:
                    return False

                # Pop atomically to prevent concurrent processing
                data = self.jobs.pop(job_id)

            # Process outside lock (optional but better)
            result = processor(data)

            # Write result
            with self.lock:
                self.results[job_id] = result

            return True