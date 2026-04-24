import uuid
from collections import deque
from typing import Dict, Any, Optional

class JobQueue:
    """
    A basic FIFO job queue implementation.
    """
    def __init__(self):
        # Queue of pending jobs (FIFO)
        # Stores tuples: (job_id, data)
        self._pending_jobs: deque[tuple[str, dict]] = deque()
        
        # Storage for completed job results
        # Stores: {job_id: result_data}
        self._job_results: Dict[str, dict] = {}
        
        # Tracks which jobs are currently being processed or waiting
        self._active_jobs: set[str] = set()

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Adds a job to the queue.
        
        Args:
            job_id: The unique identifier for the job.
            data: The data payload for the job.
            
        Returns:
            The job_id (or a generated one if the input ID was placeholder).
        """
        # Ensure job_id is unique if not provided, but requirement asks to use provided ID
        if job_id in self._active_jobs or job_id in self._job_results:
            raise ValueError(f"Job ID '{job_id}' is already active or completed.")
            
        self._pending_jobs.append((job_id, data))
        self._active_jobs.add(job_id)
        return job_id

    def get_result(self, job_id: str) -> Optional[dict]:
        """
        Retrieves the result of a completed job.
        
        Args:
            job_id: The unique identifier of the job.
            
        Returns:
            The result dictionary if available, otherwise None.
        """
        return self._job_results.get(job_id)

    def process_next_job(self) -> Optional[str]:
        """
        Simulates the processing of the next job in the queue.
        
        Returns:
            The job_id if a job was processed, otherwise None.
        """
        if not self._pending_jobs:
            return None

        job_id, data = self._pending_jobs.popleft()
        
        print(f"--- Processing Job: {job_id} with data: {data} ---")
        
        # --- SIMULATED PROCESSING LOGIC ---
        # In a real system, this would involve calling a worker function.
        # Here, we simulate a successful execution and generate a result.
        
        # Simulate work based on input data
        if data.get("task") == "process_data":
            result = {"status": "success", "output": f"Processed {data['task']} successfully."}
        else:
            result = {"status": "error", "message": "Unknown task."}
        
        # Store the result
        self._job_results[job_id] = result
        
        # Mark job as complete
        self._active_jobs.discard(job_id)
        
        return job_id

# Example Usage:
if __name__ == "__main__":
    queue = JobQueue()

    # 1. Add jobs
    job_id_1 = queue.add_job("job1", {"task": "process_data", "input_size": 100})
    job_id_2 = queue.add_job("job2", {"task": "generate_report"})
    job_id_3 = queue.add_job("job3", {"task": "process_data", "input_size": 50})

    print(f"Added job 1: {job_id_1}")
    print(f"Added job 2: {job_id_2}")
    print(f"Added job 3: {job_id_3}\n")

    # 2. Process the jobs (FIFO order: job1, job2, job3)
    print("Starting processing...")
    
    # Process job 1
    queue.process_next_job()
    
    # Process job 2
    queue.process_next_job()
    
    # Process job 3
    queue.process_next_job()
    
    print("\nProcessing complete.")

    # 3. Retrieve results
    
    # Job 1 result
    result1 = queue.get_result("job1")
    print(f"\nResult for job1: {result1}")

    # Job 2 result
    result2 = queue.get_result("job2")
    print(f"Result for job2: {result2}")

    # Try retrieving a non-existent job
    result_unknown = queue.get_result("job_unknown")
    print(f"Result for job_unknown: {result_unknown}")