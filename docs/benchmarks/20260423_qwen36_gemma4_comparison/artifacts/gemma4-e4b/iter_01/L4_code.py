import threading
import time # Added for demonstration purposes

class JobQueue:
    """
    A thread-safe job queue implementation using threading.Lock.
    """
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize the lock
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        """Adds a job safely."""
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        """
        Processes a job safely. 
        
        NOTE: We minimize the time the lock is held by extracting the job data 
        and deleting the job entry *before* running the external processor function.
        The lock is re-acquired only for the final state update (results).
        """
        job_data = None
        
        # --- Phase 1: Acquire job data and remove it (Critical Section 1) ---
        with self.lock:
            if job_id not in self.jobs:
                return False
            
            # Safely retrieve the job data
            job_data = self.jobs[job_id]
            
            # Safely mark the job as being processed by removing it
            del self.jobs[job_id]
        
        # --- Phase 2: Perform the heavy computation (Outside the lock) ---
        # This allows other threads to add/process jobs while this one is running.
        result = processor(job_data)

        # --- Phase 3: Store the result (Critical Section 2) ---
        with self.lock:
            self.results[job_id] = result
        
        return True

    def get_result(self, job_id):
        """Retrieves a result safely."""
        with self.lock:
            return self.results.get(job_id)

# --- Example Usage and Verification ---

def sample_processor(data):
    """A simulated time-consuming job processor."""
    time.sleep(0.1) # Simulate work
    return f"Processed: {data} - Time: {time.time()}"

if __name__ == '__main__':
    queue = JobQueue()
    
    # Add jobs
    job_ids = []
    for i in range(5):
        job_id = queue.add_job(f"Job_{i}", f"Data_{i}")
        job_ids.append(job_id)

    print("Jobs added. Starting processing...")

    def worker(job_id):
        success = queue.process_job(job_id, sample_processor)
        if success:
            result = queue.get_result(job_id)
            print(f"[{threading.current_thread().name}] Completed {job_id}. Result: {result[:30]}...")
        else:
            print(f"[{threading.current_thread().name}] Failed to find {job_id}.")

    threads = []
    for job_id in job_ids:
        t = threading.Thread(target=worker, args=(job_id,), name=f"Worker-{job_id}")
        threads.append(t)
        t.start()

    # Wait for all threads to finish
    for t in threads:
        t.join()

    # Verify state
    print("\n--- Verification ---")
    print(f"Jobs remaining in queue: {len(queue.jobs)}") # Should be 0
    print(f"Results stored: {len(queue.results)}")     # Should be 5