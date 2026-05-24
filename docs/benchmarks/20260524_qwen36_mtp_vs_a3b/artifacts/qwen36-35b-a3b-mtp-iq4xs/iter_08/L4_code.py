import threading

    class JobQueue:
        def __init__(self):
            self.jobs = {}
            self.results = {}
            self.lock = threading.Lock() # 1. Create lock

        def add_job(self, job_id, data):
            with self.lock: # 2. Protect add_job too for thread safety
                self.jobs[job_id] = data
            return job_id

        def process_job(self, job_id, processor):
            with self.lock: # 3. Acquire lock
                if job_id not in self.jobs:
                    return False
                
                # We must remove the job immediately to prevent other threads
                # from picking it up while we process it.
                data = self.jobs.pop(job_id) 
            
            # 4. Process outside lock to allow concurrency for different jobs
            result = processor(data)
            
            # 5. Lock again to write result safely (or just write, but lock is safer)
            with self.lock:
                self.results[job_id] = result
            
            return True