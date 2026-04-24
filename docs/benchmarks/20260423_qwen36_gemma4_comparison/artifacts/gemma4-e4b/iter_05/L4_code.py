import threading
import time
import random

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize the lock to protect shared resources
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        """Adds a job to the queue."""
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        """
        Processes a job. The entire critical section (checking existence, 
        retrieving data, updating results, and deleting the job) must be 
        protected by the lock.
        """
        data = None
        
        # Phase 1: Acquire lock to safely retrieve and remove the job
        with self._lock:
            if job_id not in self.jobs:
                return False
            
            data = self.jobs[job_id]
            # Remove the job immediately to prevent other threads from picking it up
            del self.jobs[job_id] 

        # Phase 2: Execute the job (outside the lock if processor is long-running)
        # Note: If the processor is very fast, keeping it inside the lock is simpler, 
        # but releasing the lock allows other threads to add/retrieve jobs while
        # the CPU is busy with the processing task.
        try:
            result = processor(data)
        except Exception as e:
            print(f"Job {job_id} failed: {e}")
            return False

        # Phase 3: Acquire lock again to safely store the result
        with self._lock:
            self.results[job_id] = result
            
        return True

    def get_result(self, job_id):
        """Retrieves the result for a job, safely."""
        with self._lock:
            return self.results.get(job_id)

# --- Example Usage and Verification ---

def dummy_processor(data):
    """A simulated job processor function."""
    # Simulate work being done
    time.sleep(random.uniform(0.01, 0.05))
    return f"Processed: {data} successfully"

if __name__ == '__main__':
    queue = JobQueue()
    job_ids = []
    NUM_JOBS = 10

    # Add jobs
    for i in range(NUM_JOBS):
        job_id = queue.add_job(f"job_{i}", f"data_{i}")
        job_ids.append(job_id)

    print(f"--- Starting processing {NUM_JOBS} jobs ---")
    
    threads = []
    
    # Create threads to process jobs concurrently
    def worker():
        # Threads will randomly try to process jobs until all are done
        for job_id in job_ids:
            if queue.process_job(job_id, dummy_processor):
                # print(f"Thread processed {job_id}")
                pass
            time.sleep(0.001) # Small delay to simulate real-world contention

    # Start multiple worker threads
    for _ in range(5):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()

    # Wait for all threads to finish
    for t in threads:
        t.join()

    print("\n--- Verification ---")
    
    # Check if all jobs were processed and results are available
    success_count = 0
    for job_id in job_ids:
        result = queue.get_result(job_id)
        if result:
            print(f"Result for {job_id}: {result}")
            success_count += 1
        else:
            print(f"ERROR: Result not found for {job_id}")

    print(f"\nTotal jobs processed: {success_count}/{NUM_JOBS}")
    print(f"Jobs remaining in queue: {len(queue.jobs)}")