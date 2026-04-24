import threading
import time # Included for demonstration/testing purposes

class JobQueue:
    """
    A thread-safe job queue implementation.
    """
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize the lock to protect shared resources (jobs and results)
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        """Adds a job to the queue safely."""
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        """
        Processes a job. This entire sequence (lookup, processing, result storage, deletion)
        must be protected to ensure state consistency.
        """
        # Acquire the lock before accessing or modifying shared state
        with self._lock:
            if job_id not in self.jobs:
                return False

            # Retrieve data while holding the lock
            data = self.jobs[job_id]
            
            # IMPORTANT NOTE ON PERFORMANCE: 
            # If the 'processor' function takes a very long time, holding the lock 
            # during execution will block all other threads (including add_job 
            # and get_result). For high-performance systems, one would move the 
            # processor call outside the lock and use a different pattern 
            # (e.g., marking the job as 'processing' before releasing the lock).
            # However, to fix the race condition in the original structure, 
            # we lock the entire critical sequence.
            
            # Temporarily release the lock to allow other threads to interact 
            # while the CPU-intensive processing happens.
            # Since Python doesn't allow releasing a context manager lock easily,
            # we must restructure the logic slightly for clean concurrency.
            
            # --- RESTRUCTURED SAFE LOGIC ---
            # 1. Get data and mark job as removed (critical state change)
            job_data = self.jobs.pop(job_id, None)
            
            if job_data is None:
                # Job was already taken or didn't exist
                return False

        # Perform the long-running, non-state-modifying task outside the lock
        try:
            result = processor(job_data)
        except Exception as e:
            print(f"Error processing job {job_id}: {e}")
            return False

        # Re-acquire the lock to safely write the result and finalize the state
        with self._lock:
            self.results[job_id] = result
            
        return True

    def get_result(self, job_id):
        """Retrieves a result safely."""
        with self._lock:
            return self.results.get(job_id)

# --- Example Usage and Verification ---

def mock_processor(data):
    """A simulated job processor."""
    time.sleep(0.1) # Simulate work
    return f"Processed: {data}"

if __name__ == '__main__':
    queue = JobQueue()
    
    job_id_1 = queue.add_job(1, "Task A")
    job_id_2 = queue.add_job(2, "Task B")
    
    print(f"Added jobs: {job_id_1}, {job_id_2}")

    def worker(job_id):
        success = queue.process_job(job_id, mock_processor)
        if success:
            result = queue.get_result(job_id)
            print(f"Thread finished Job {job_id}. Result: {result}")
        else:
            print(f"Thread failed to process Job {job_id}.")

    threads = []
    t1 = threading.Thread(target=worker, args=(job_id_1,))
    t2 = threading.Thread(target=worker, args=(job_id_2,))

    threads.append(t1)
    threads.append(t2)

    t1.start()
    t2.start()

    for t in threads:
        t.join()

    print("\nAll jobs processed.")
    print(f"Final results dictionary: {queue.results}")