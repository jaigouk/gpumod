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
            # We need to check and "claim" the job atomically
            with self.lock:
                if job_id not1 in self.jobs:
                    return False
                # To prevent another thread from picking this up 
                # while we are processing, we can mark it or 
                # remove it immediately.
                # Removing it immediately is the cleanest way to 
                # "claim" it.
                data = self.jobs.pop(job_id)
            
            # Now we process outside the lock so other jobs can be processed
            try:
                result = Processor(data)
                with self.lock:
                    self.results[job_id] = result
                return True
            except Exception:
                # Optional: handle errors if needed
                return False

        def get_result(self, job_id):
            with self.lock:
                return self.results.get(job_id)