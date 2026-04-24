import threading

class JobQueue:
    """
    A thread-safe job queue implementation using a threading.Lock.
    """
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize the lock to protect shared state
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        """Adds a job to the queue in a thread-safe manner."""
        with self._lock:
            self.jobs[job_id] = data
            return job_id

    def process_job(self, job_id, processor):
        """
        Processes a job. This is the critical section that must be protected.
        """
        # Acquire the lock before accessing shared state
        with self._lock:
            if job_id not in self.jobs:
                return False

            # Get data while holding the lock
            data = self.jobs[job_id]
            
            # Release the lock temporarily while performing the potentially long-running task
            # NOTE: For maximum concurrency, we should release the lock *before* calling 
            # the processor if the processor itself is slow. However, since the prompt 
            # implies the race condition is around dictionary modification, 
            # we will keep the critical modifications atomic. 
            # For simplicity and strict atomicity of the entire process_job operation:
            
            try:
                # Perform the processing outside the lock if it's slow, 
                # but for this simple fix, we keep the logic atomic.
                result = processor(data)
            except Exception as e:
                # Handle processing errors if necessary
                print(f"Job {job_id} failed: {e}")
                return False

            # Write result and delete job while holding the lock
            self.results[job_id] = result
            del self.jobs[job_id]
            
            return True

    def get_result(self, job_id):
        """Retrieves a result in a thread-safe manner."""
        # Acquire the lock even for reading to ensure we don't read during a modification
        with self._lock:
            return self.results.get(job_id)

# --- Example Usage and Verification ---

def simple_processor(data):
    import time
    time.sleep(0.01) # Simulate work
    return f"Processed: {data}"

if __name__ == '__main__':
    queue = JobQueue()
    
    # Add jobs
    job_ids = []
    for i in range(5):
        job_id = queue.add_job(f"job_{i}", f"data_{i}")
        job_ids.append(job_id)

    def worker(job_id):
        # Simulate multiple workers trying to process jobs concurrently
        success = queue.process_job(job_id, simple_processor)
        if success:
            print(f"Worker finished {job_id}")
        else:
            print(f"Worker failed to process {job_id}")

    threads = []
    for job_id in job_ids:
        t = threading.Thread(target=worker, args=(job_id,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Check results
    print("\n--- Final Results ---")
    for job_id in job_ids:
        result = queue.get_result(job_id)
        print(f"{job_id}: {result}")
    
    # Check if jobs queue is empty
    print(f"\nJobs remaining in queue: {queue.jobs}")