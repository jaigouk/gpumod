import collections
import time
from typing import Dict, Any, Optional

class JobQueue:
    """
    A basic job queue implementation for managing and retrieving job results.
    """
    def __init__(self):
        # Queue holding job IDs in FIFO order
        self._job_queue = collections.deque()
        
        # Stores job data for pending jobs (job_id -> data)
        self._pending_jobs: Dict[str, Dict[str, Any]] = {}
        
        # Stores results of completed jobs (job_id -> result_data)
        self._results: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Adds a job to the queue.
        """
        if job_id in self._pending_jobs or job_id in self._results:
            raise ValueError(f"Job ID '{job_id}' already exists.")
            
        self._pending_jobs[job_id] = data
        self._job_queue.append(job_id)
        return job_id

    def _simulate_job_processing(self, job_id: str, job_data: dict) -> dict:
        """
        Internal method to simulate the execution of a job.
        In a real system, this would involve calling worker functions.
        """
        print(f"--- Processing Job: {job_id} with data: {job_data} ---")
        # Simulate work time
        time.sleep(0.1) 
        
        # Simulate result generation based on input data
        task = job_data.get("task", "unknown")
        if task == "process_data":
            result = {"status": "completed", "output": f"Processed data successfully for {job_id}"}
        else:
            result = {"status": "error", "message": "Unknown task"}
            
        return result

    def process_next_job(self) -> Optional[str]:
        """
        Retrieves the next job from the queue and processes it.
        Returns the job_id if processed, otherwise None.
        """
        if not self._job_queue:
            return None

        job_id = self._job_queue.popleft()
        
        if job_id not in self._pending_jobs:
            # Should not happen if state is managed correctly
            print(f"Warning: Job ID {job_id} found in queue but not in pending jobs.")
            return None

        job_data = self._pending_jobs.pop(job_id)
        
        try:
            result = self._simulate_job_processing(job_id, job_data)
            self._results[job_id] = result
            return job_id
        except Exception as e:
            print(f"Error processing job {job_id}: {e}")
            # Handle failure scenario (e.g., store error status in results)
            self._results[job_id] = {"status": "failed", "error": str(e)}
            return job_id

    def get_result(self, job_id: str) -> Optional[dict]:
        """
        Retrieves the result of a completed job.
        Returns None if the job ID is not found or not yet completed.
        """
        return self._results.get(job_id)

# Example Usage Demonstration:
if __name__ == '__main__':
    queue = JobQueue()

    # 1. Add jobs
    job_id_1 = queue.add_job("job1", {"task": "process_data", "payload": 10})
    job_id_2 = queue.add_job("job2", {"task": "calculate_sum", "payload": 5})
    job_id_3 = queue.add_job("job3", {"task": "process_data", "payload": 20})

    print(f"Added job 1: {job_id_1}")
    print(f"Added job 2: {job_id_2}")
    print(f"Added job 3: {job_id_3}\n")

    # 2. Attempt to get results immediately (should be None)
    print("--- Checking results before processing ---")
    print(f"Result for job1 (initial): {queue.get_result('job1')}")

    # 3. Process jobs sequentially (FIFO order: job1, job2, job3)
    print("\n--- Starting Job Processing ---")
    
    # Process Job 1
    queue.process_next_job()
    
    # Process Job 2
    queue.process_next_job()

    # Process Job 3
    queue.process_next_job()
    
    print("\n--- Processing Complete ---")

    # 4. Retrieve results
    print("\n--- Retrieving Results ---")
    result1 = queue.get_result("job1")
    print(f"Result for job1: {result1}")

    result3 = queue.get_result("job3")
    print(f"Result for job3: {result3}")
    
    # Check for a non-existent job
    result_missing = queue.get_result("job_missing")
    print(f"Result for job_missing: {result_missing}")