import threading

class JobQueue:
    """
    A thread-safe job queue implementation using threading.Lock.
    """
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize the lock to synchronize access to self.jobs and self.results
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        """Adds a job to the queue safely."""
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        """
        Processes a job. This operation must be atomic:
        (Check existence -> Retrieve data -> Remove from jobs -> Store result).
        """
        data = None
        
        # Phase 1: Critical section for state change (Retrieval and Deletion)
        with self._lock:
            if job_id not in self.jobs:
                return False
            
            # Retrieve data and immediately remove it from the pending queue
            data = self.jobs[job_id]
            del self.jobs[job_id]
        
        # Phase 2: External computation (Ideally done outside the lock 
        # to prevent blocking other queue operations)
        result = processor(data)

        # Phase 3: Critical section for state change (Storing result)
        with self._lock:
            self.results[job_id] = result
            
        return True

    def get_result(self, job_id):
        """Safely retrieves a job result."""
        with self._lock:
            return self.results.get(job_id)

# Example Usage and Verification (Optional)
if __name__ == '__main__':
    import time
    import random

    queue = JobQueue()

    def worker(job_id, data):
        # Simulate processing time
        time.sleep(random.uniform(0.1, 0.5))
        return f"Processed {data} successfully"

    # Add jobs
    job_ids = []
    for i in range(5):
        job_id = queue.add_job(f"job_{i}", f"data_{i}")
        job_ids.append(job_id)
    
    print("Jobs added.")

    # Simulate processing jobs concurrently
    def job_runner(job_id):
        success = queue.process_job(job_id, worker)
        if success:
            print(f"Thread processed {job_id}")

    threads = []
    for job_id in job_ids:
        t = threading.Thread(target=job_runner, args=(job_id,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("\n--- Results ---")
    for job_id in job_ids:
        result = queue.get_result(job_id)
        print(f"{job_id}: {result}")

    # Verify that the queue is empty
    with queue._lock:
        print(f"\nPending jobs remaining: {len(queue.jobs)}")