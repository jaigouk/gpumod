import threading

    class JobQueue:
        def __init__(self):
            self.jobs = {}
            self.results = {}
            self.lock = threading.Lock() # NEW

        def add_job(self, job_id, data):
            with self.lock: # PROTECT
                self.jobs[job_id] = data
                return job_id

        def process_job(self, job_id, processor):
            with self.lock: # PROTECT
                if job_id not in self.jobs:
                    return False

                data = self.jobs[job_id]
                # Note: The actual processing might take time.
                # Ideally, we shouldn't hold the lock during heavy processing
                # if we want high concurrency, but for a simple queue fix,
                # we usually lock the state transitions.
                # However, the prompt asks to fix the race condition on the
                # shared data structures.
                
                result = processor(data) 
                # CRITICAL: If processor is CPU intensive, holding the lock
                # blocks other threads. A better design extracts data first,
                # releases lock, processes, then acquires lock to write result.
                # But strictly speaking, the bug is in the dictionary access.
                
                self.results[job_id] = result
                del self.jobs[job_id]
                return True

        def get_result(self, job_id):
            with self.lock: # PROTECT
                return self.results.get(job_id)