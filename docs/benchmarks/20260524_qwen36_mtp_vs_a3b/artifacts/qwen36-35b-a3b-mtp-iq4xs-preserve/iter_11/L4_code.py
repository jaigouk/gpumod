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
            # Critical to hold lock during check and modification
            with self.lock:
                if job_id not in self.jobs:
                    return False

                data = self.jobs[job_id]
                # NOTE: Processing (processor(data)) usually takes time.
                # If we hold the lock here, we block all other threads.
                # Optimization: Copy data out, release lock, process, re-acquire lock.
                # However, strictly speaking, to fix the *race condition* described
                # (concurrent writes to results), the lock is needed.
                # But holding the lock during CPU-bound processing is bad practice.
                # Let's look at the specific bug line: self.results[job_id] = result.
                # And the deletion: del self.jobs[job_id].

                # Let's do the safe way (release lock during processing):
                del self.jobs[job_id] # Remove from jobs so others don't pick it up
                # Release lock now?
            # Processing happens outside lock
            result = processor(data)

            # Now we need to save the result.
            with self.lock:
                self.results[job_id] = result
            return True

        def get_result(self, job_id):
            with self.lock:
                return self.results.get(job_id)