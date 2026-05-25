import threading

    class JobQueue:
        def __init__(self):
            self.jobs = {}
            self.results = {}
            self.lock = threading.Lock() # 1. Create lock

        def add_job(self, job_id, data):
            with self.lock: # 2. Lock
                self.jobs[job_id] = data
            # Return ID? Prompt code didn't return ID, but logic implies it. 
            # Actually prompt code: "return job_id". Okay.

        def process_job(self, job_id, processor):
            with self.lock: # 3. Lock
                if job_id not in self.jobs:
                    return False
                
                # Atomic check and retrieve
                data = self.jobs[job_id]
                del self.jobs[job_id] # Remove so others don't grab it
            
            # 4. Process outside lock (good practice, though strict fix might keep it inside)
            # However, for a simple "Fix the bug" task, keeping it inside is safer 
            # to prevent the dictionary from changing state unexpectedly, 
            # but holding a lock during 'processor(data)' is bad design if processor blocks.
            # Let's check the prompt constraints. It's a "bug fix".
            # If I process inside the lock, it's safe but slow.
            # If I process outside, I need to be careful.
            
            # Let's look at the specific line flagged: self.results[job_id] = result.
            # If I move processing outside, I must lock again for the result write.
            
            result = processor(data)
            
            with self.lock: # 5. Lock again to write result
                self.results[job_id] = result
            
            return True

        def get_result(self, job_id):
            with self.lock: # 6. Lock for read consistency
                return self.results.get(job_id)