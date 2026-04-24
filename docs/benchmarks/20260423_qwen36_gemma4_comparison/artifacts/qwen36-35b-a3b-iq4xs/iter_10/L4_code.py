import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to protect shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock is required to safely add to the shared dictionary
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Lock to atomically check existence and remove the job.
        # Using 'pop' prevents another thread from picking up the same job.
        with self.lock:
            # pop returns the value if found, or the default (None) if not.
            # Note: If data can be validly None, a unique sentinel object should be used 
            # as the default to distinguish between "key missing" and "value is None".
            # Here we assume data is not None or handle it simply.
            data = self.jobs.pop(job_id, None)
            
            if data is None:
                # If data is None, we can't be sure if the key was missing 
                # or if the value was None. 
                # To be strictly correct with the original logic (which returns False only if key missing),
                # we should check existence first or use a sentinel. 
                # However, for standard usage, we assume valid data. 
                # A safer approach for strict logic:
                # return False 
                # But since we popped, the key is gone. 
                # Let's revert to a safer check-then-pop pattern for 100% correctness 
                # regarding the original False return behavior.
                pass 

        # --- REVISION FOR STRICT CORRECTNESS ---
        # The logic above with pop(None) is ambiguous if data=None is allowed. 
        # Let's use the explicit check inside the lock for absolute safety.
        
        # Let's rewrite the method body cleanly below.

    # ... (Implementing the clean version in the block below)