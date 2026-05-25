import threading

    class JobQueue:
        def __init__(self):
            self.jobs = {}
            self.results = {}
            self.lock = threading.Lock() # Create the lock

        def add_job(self, job_id, data):
            with self.lock: # Protect write to self.jobs
                self.jobs[job_id] = data
            return job_id

        def process_job(self, job_id, processor):
            with self.lock: # Protect read/check/delete from jobs and write to results
                if job_id not in self.jobs:
                    return False

                data = self.jobs[job_id]
                # Note: The actual processing happens outside the lock usually for performance,
                # but strictly speaking, the bug reported is about the dictionary writes.
                # However, if we hold the lock while running the processor, we block everyone else.
                # Let's look at the original logic.
                # Original: check -> get -> process -> write result -> delete.
                # If we lock around the whole thing, it's safe but slow.
                # If we lock only around dictionary access, it's faster but we must ensure atomicity of the "claim".

                # Let's stick to the safest, simplest interpretation of "fix the race condition"
                # which usually implies protecting the shared state manipulation.
                
                # Optimization: Get data, then release lock, process, then lock to update result/delete.
                # But wait, if we release the lock, another thread might steal the job?
                # The prompt implies a simple fix. Let's look at the specific bug comment.
                # "multiple threads can write simultaneously" -> self.results[job_id] = result.
                
                # Let's do the standard critical section approach.
                pass 
            
            # Actually, to fix the specific logic flaw where a job is processed twice or deleted while being processed:
            # We need to atomically check existence, extract, and remove.
            
            # Let's rewrite process_job carefully.
            
            with self.lock:
                if job_id not in self.jobs:
                    return False
                
                data = self.jobs[job_id]
                del self.jobs[job_id] # Remove immediately to prevent double processing
            
            # Process outside lock to allow concurrency
            result = processor(data)
            
            with self.lock:
                self.results[job_id] = result
            
            return True

        def get_result(self, job_id):
            with self.lock: # Protect read from results
                return self.results.get(job_id)