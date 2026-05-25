import threading

    class JobQueue:
        def __init__(self):
            self.jobs = {}
            self.results = {}
            self.lock = threading.Lock() # 1. Initialize Lock

        def add_job(self, job_id, data):
            with self.lock: # 2. Lock add_job
                self.jobs[job_id] = data
            return job_id

        def process_job(self, job_id, processor):
            with self.lock: # 3. Lock the critical section
                # Check and remove atomically to prevent double processing
                if job_id not in self.jobs:
                    return False
                
                data = self.jobs[job_id]
                # Remove immediately so other threads don't pick it up
                del self.jobs[job_id] 
            
            # 4. Process outside lock (optional but better for performance)
            # BUT, the original code structure implies process_job is the transaction.
            # If we process outside, we must be careful.
            # Let's stick to the simplest fix that guarantees correctness based on the prompt's focus.
            
            result = processor(data)
            
            with self.lock: # 5. Lock for writing result
                self.results[job_id] = result
                
            return True