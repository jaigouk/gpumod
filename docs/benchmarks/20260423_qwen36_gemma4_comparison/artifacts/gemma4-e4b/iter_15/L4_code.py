import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize the lock to protect shared state
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        # Lock ensures atomic insertion into the jobs dictionary
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        data = None
        
        # Step 1: Check existence and retrieve data atomically
        # We must lock here to ensure the job exists and to safely retrieve/remove it.
        with self._lock:
            if job_id not in self.jobs:
                return False
            
            # Retrieve the data and immediately remove the job from the queue.
            # This prevents other threads from processing the same job.
            data = self.jobs[job_id]
            del self.jobs[job_id]

        # Step 2: Process the job (CPU intensive work done outside the lock 
        # to minimize contention and improve performance)
        try:
            result = processor(data)
        except Exception as e:
            # Handle processing errors gracefully
            print(f"Error processing job {job_id}: {e}")
            return False

        # Step 3: Store the result atomically
        with self._lock:
            self.results[job_id] = result
        
        return True

    def get_result(self, job_id):
        # Lock ensures that we read a consistent state of the results dictionary
        with self._lock:
            return self.results.get(job_id)

# --- Example Usage (Demonstration) ---

def simple_processor(data):
    """A dummy function simulating job processing."""
    import time
    time.sleep(0.01) # Simulate work
    return f"Processed: {data}"

if __name__ == '__main__':
    queue = JobQueue()
    
    # Add jobs
    job_ids = [queue.add_job(f"job_{i}", f"data_{i}") for i in range(5)]
    print(f"Jobs added: {job_ids}")
    
    threads = []
    
    # Start multiple threads trying to process jobs concurrently
    def worker(job_id, processor):
        success = queue.process_job(job_id, processor)
        if success:
            print(f"Thread processed {job_id} successfully.")

    for job_id in job_ids:
        t = threading.Thread(target=worker, args=(job_id, simple_processor))
        threads.append(t)
        t.start()

    # Wait for all processing threads to finish
    for t in threads:
        t.join()

    print("\n--- Final State Check ---")
    
    # Check results
    for job_id in job_ids:
        result = queue.get_result(job_id)
        print(f"{job_id} result: {result}")

    # Check if jobs queue is empty
    with queue._lock:
        print(f"Jobs remaining in queue: {len(queue.jobs)}")