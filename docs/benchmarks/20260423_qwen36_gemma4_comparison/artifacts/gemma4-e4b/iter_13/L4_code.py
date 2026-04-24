import threading
import time
import random

class JobQueue:
    """
    A thread-safe implementation of a job queue using a threading.Lock.
    """
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize the lock to synchronize access to shared resources
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        """Adds a job to the queue in a thread-safe manner."""
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        """
        Retrieves, processes, and stores the result for a job in a thread-safe manner.
        The entire sequence (check, read, write, delete) must be atomic.
        """
        data = None
        
        # 1. Atomically retrieve and remove the job data
        with self._lock:
            if job_id not in self.jobs:
                return False
            
            # Use pop() to retrieve and remove the item in one atomic operation
            data = self.jobs.pop(job_id, None) 
        
        if data is None:
            # Should only happen if another thread processed it between checks
            return False

        # 2. Execute the long-running processor (outside the lock to prevent blocking other threads)
        # NOTE: If the processor function itself modifies shared state, it must handle its own locking.
        try:
            result = processor(data)
        except Exception as e:
            print(f"Error processing job {job_id}: {e}")
            return False
        
        # 3. Atomically store the result
        with self._lock:
            self.results[job_id] = result
            
        return True

    def get_result(self, job_id):
        """Retrieves a result in a thread-safe manner."""
        with self._lock:
            return self.results.get(job_id)

# --- Example Usage and Verification ---

def simple_processor(data):
    """A simulated job processor."""
    # Simulate work being done
    time.sleep(random.uniform(0.01, 0.1)) 
    return f"Processed data: {data * 2}"

def worker(queue, job_id):
    """A thread worker that attempts to process a specific job."""
    print(f"Worker {job_id} attempting to process job...")
    success = queue.process_job(job_id, simple_processor)
    if success:
        result = queue.get_result(job_id)
        print(f"Worker {job_id} successfully finished job. Result: {result}")
    else:
        print(f"Worker {job_id} failed to process job (already gone or nonexistent).")


if __name__ == '__main__':
    queue = JobQueue()
    
    # Add jobs
    job_ids = [queue.add_job(f'job_{i}', f'input_{i}') for i in range(5)]
    print(f"Jobs added: {job_ids}")

    threads = []
    
    # Create multiple threads to race for the jobs
    # We will create 7 threads, but only 5 jobs exist, forcing contention.
    for i, job_id in enumerate(job_ids):
        t = threading.Thread(target=worker, args=(queue, job_id))
        threads.append(t)
        t.start()
    
    # Start a few extra threads that will fail gracefully
    for i in range(5, 7):
        t = threading.Thread(target=worker, args=(queue, f'job_{i}'))
        threads.append(t)
        t.start()


    # Wait for all threads to complete
    for t in threads:
        t.join()

    print("\n--- Final State Check ---")
    print(f"Jobs remaining in queue: {queue.jobs}")
    print(f"Results available: {len(queue.results)}")
    
    # Verify all 5 initial jobs were processed exactly once
    assert len(queue.results) == 5
    assert len(queue.jobs) == 0
    print("Verification successful: All jobs were processed exactly once.")