import threading
import time # Added for demonstration/testing purposes

class JobQueue:
    """
    A thread-safe job queue implementation using threading.Lock.
    """
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to protect shared resources
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        """Adds a job to the queue safely."""
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        """
        Pulls a job, processes it, and stores the result safely.
        The entire sequence (check, read, write, delete) is atomic.
        """
        
        # Step 1: Check existence and retrieve data under lock
        with self._lock:
            if job_id not in self.jobs:
                return False

            # Retrieve the data while holding the lock
            data = self.jobs[job_id]
            
            # Delete the job immediately to prevent other threads from processing it
            # Note: We must release the lock before running the potentially long-running 
            # processor to avoid blocking the entire queue.
            del self.jobs[job_id]

        # Step 2: Process the job (outside the lock for better concurrency)
        # WARNING: If the processor function itself modifies the queue state, 
        # it must handle its own locking.
        result = processor(data)

        # Step 3: Store the result under lock
        with self._lock:
            self.results[job_id] = result
        
        return True

    def get_result(self, job_id):
        """Retrieves a result safely."""
        with self._lock:
            return self.results.get(job_id)

# ======================================================================
# Example Usage (Demonstration of thread safety)
# ======================================================================

def worker_function(queue, job_id, job_data):
    """Simulates a worker thread processing a job."""
    print(f"Thread {threading.get_ident()} started job {job_id}")
    
    # Simulate processing time
    time.sleep(0.1) 
    
    # The queue handles the actual processing and result storage
    success = queue.process_job(job_id, lambda data: f"Processed: {data} by thread {threading.get_ident()}")
    
    if success:
        print(f"Thread {threading.get_ident()} finished job {job_id} successfully.")
    else:
        print(f"Thread {threading.get_ident()} failed to find job {job_id}.")


if __name__ == '__main__':
    queue = JobQueue()
    threads = []
    
    # Add jobs
    job_ids = ['J1', 'J2', 'J3', 'J4']
    for i, job_id in enumerate(job_ids):
        queue.add_job(job_id, f"Data for job {i}")

    print("Jobs added to the queue.")

    # Start multiple threads trying to process the same jobs (or different ones)
    for job_id in job_ids:
        t = threading.Thread(target=worker_function, args=(queue, job_id, None))
        threads.append(t)
        t.start()

    # Wait for all threads to finish
    for t in threads:
        t.join()

    print("\n--- Final State ---")
    # Check results
    for job_id in job_ids:
        result = queue.get_result(job_id)
        print(f"Result for {job_id}: {result}")
        
    # Check if all jobs were removed from the active queue
    print(f"Jobs remaining in queue: {list(queue.jobs.keys())}")