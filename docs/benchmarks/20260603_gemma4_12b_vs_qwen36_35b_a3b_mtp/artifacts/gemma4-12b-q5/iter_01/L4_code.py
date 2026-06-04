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
            # Use a lock to ensure check and removal/result-writing is atomic
            # However, we shouldn't hold the lock during the actual processing 
            # to allow multiple jobs to run in parallel.
            
            data = None
            with self.lock:
                if job_id not in self.jobs:
                    return False
                # Remove it from jobs immediately so no other thread picks it up
                data = self.jobs.pop(job_id)

            result = processor(data)

            with self.lock:
                self.results[job_id] = result
            return True