import threading

    class JobQueue:
        def __init__(self):
            self.jobs = {}
            self.results = {}
            self.lock = threading.Lock() # 1. Initialize lock

        def add_job(self, job_id, data):
            with self.lock: # 2. Lock access
                self.jobs[job_id] = data
            return job_id

        def process_job(self, job_id, processor):
            with self.lock: # 3. Lock access
                # Check existence
                if job_id not in self.jobs:
                    return False
                
                # Get data
                data = self.jobs[job_id]
                
                # CRITICAL: Do we process inside the lock?
                # If processor is slow, holding the lock blocks other threads.
                # However, we need to ensure we don't process the same job twice.
                # To prevent double processing, we must remove the job from 'jobs' 
                # BEFORE calling the processor, but inside the lock.
                
                del self.jobs[job_id] # Remove immediately to claim the job
            
            # Now process outside the lock to allow concurrency
            result = processor(data)
            
            # Write result. Is this thread safe?
            # Writing to a dict is atomic in CPython, but for safety/consistency:
            with self.lock:
                self.results[job_id] = result
            return True

        def get_result(self, job_id):
            with self.lock: # 4. Lock access
                return self.results.get(job_id)