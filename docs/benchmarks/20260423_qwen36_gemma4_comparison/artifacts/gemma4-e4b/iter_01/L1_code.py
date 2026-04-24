import time
import uuid
from typing import Dict, Any, Optional, Deque
from collections import deque

class JobQueue:
    """
    A basic FIFO job queue implementation.
    """
    def __init__(self):
        # Stores jobs waiting to be processed (FIFO order)
        # Format: job_id -> data
        self._pending_jobs: Deque[tuple[str, Dict[str, Any]]] = deque()
        
        # Stores the results of completed jobs
        # Format: job_id -> result_data
        self._job_results: Dict[str, Dict[str, Any]] = {}
        
        # Tracks the state of all jobs
        # Format: job_id -> state ('pending', 'completed')
        self._job_states: Dict[str, str] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> str:
        """
        Adds a job to the queue.
        
        Args:
            job_id: The unique identifier for the job.
            data: The data associated with the task.
            
        Returns:
            The job_id.
        """
        if job_id in self._job_states:
            raise ValueError(f"Job ID '{job_id}' already exists.")
            
        self._pending_jobs.append((job_id, data))
        self._job_states[job_id] = 'pending'
        return job_id

    def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the result of a completed job.
        
        Args:
            job_id: The ID of the job.
            
        Returns:
            The result dictionary if completed, otherwise None.
        """
        if self._job_states.get(job_id) == 'completed':
            return self._job_results.get(job_id)
        return None

    def process_next_job(self) -> bool:
        """
        Simulates a worker picking up and processing the next job in the queue.
        
        Returns:
            True if a job was processed, False if the queue was empty.
        """
        if not self._pending_jobs:
            return False

        job_id, data = self._pending_jobs.popleft()
        self._job_states[job_id] = 'processing'
        
        # --- Simulated Job Execution ---
        print(f"\n[Worker] Starting job: {job_id} with data: {data}")
        
        # Simulate work being done
        time.sleep(0.1) 
        
        # Simulate generating a result based on the input data
        try:
            result = {
                "status": "success",
                "processed_data": f"Processed {data.get('task', 'unknown')} for {job_id}",
                "input_data": data
            }
            
            # Store the result and update state
            self._job_results[job_id] = result
            self._job_states[job_id] = 'completed'
            print(f"[Worker] Finished job: {job_id}")
            return True
        except Exception as e:
            # Handle job failures
            self._job_states[job_id] = 'failed'
            print(f"[Worker] Job {job_id} failed: {e}")
            return True # Job was processed, even if it failed

# --- Example Usage ---
if __name__ == "__main__":
    queue = JobQueue()
    
    # 1. Add jobs
    job_id_1 = queue.add_job("job1", {"task": "process_data", "priority": 1})
    job_id_2 = queue.add_job("job2", {"task": "cleanup_files", "priority": 2})
    job_id_3 = queue.add_job("job3", {"task": "generate_report", "priority": 3})

    print(f"Added jobs: {job_id_1}, {job_id_2}, {job_id_3}")
    
    # 2. Check results before processing (should be None)
    print("\n--- Initial Check ---")
    print(f"Result for job1 (before processing): {queue.get_result(job_id_1)}")

    # 3. Process the first job (FIFO)
    print("\n--- Processing Cycle 1 ---")
    queue.process_next_job() # Processes job1

    # 4. Check result after first processing
    print("\n--- Check After Job 1 ---")
    result_1 = queue.get_result(job_id_1)
    print(f"Result for job1: {result_1}")

    # 5. Process the second job
    print("\n--- Processing Cycle 2 ---")
    queue.process_next_job() # Processes job2
    
    # 6. Check result for job2
    print("\n--- Check After Job 2 ---")
    result_2 = queue.get_result(job_id_2)
    print(f"Result for job2: {result_2}")
    
    # 7. Process the last job
    print("\n--- Processing Cycle 3 ---")
    queue.process_next_job() # Processes job3
    
    # 8. Final check
    print("\n--- Final Check ---")
    result_3 = queue.get_result(job_id_3)
    print(f"Result for job3: {result_3}")
    
    # 9. Attempt to process when empty
    print("\n--- Empty Queue Check ---")
    if not queue.process_next_job():
        print("Queue is empty.")