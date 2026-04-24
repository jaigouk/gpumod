import uuid
from collections import deque
from typing import Dict, Optional, Any

class JobQueue:
    """
    A basic in-memory job queue implementing FIFO processing.
    """
    def __init__(self):
        # Stores jobs waiting to be processed (FIFO)
        # Format: deque of (job_id, data)
        self._pending_jobs: deque[tuple[str, Dict[str, Any]]] = deque()
        
        # Stores results of completed jobs
        # Format: {job_id: result_data}
        self._completed_jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> str:
        """
        Adds a job to the queue.

        Args:
            job_id: The unique identifier for the job.
            data: The data/payload for the job.

        Returns:
            The job_id.
        """
        if not job_id:
            job_id = str(uuid.uuid4())
        
        self._pending_jobs.append((job_id, data))
        return job_id

    def process_next_job(self) -> Optional[str]:
        """
        Simulates processing the next job in the queue. 
        In a real system, this would be handled by a worker thread.

        Returns:
            The job_id if a job was processed, None otherwise.
        """
        if not self._pending_jobs:
            return None

        job_id, data = self._pending_jobs.popleft()
        
        # --- Simulation of Job Execution ---
        print(f"--- Processing job: {job_id} with data: {data} ---")
        
        # Simulate work being done
        if data.get("task") == "process_data":
            result = {"status": "success", "processed_data": f"Processed {data.get('task')}"}
        else:
            result = {"status": "failed", "error": "Unknown task"}

        # Store the result
        self._completed_jobs[job_id] = result
        print(f"--- Job {job_id} completed. ---")
        return job_id

    def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the result of a completed job.

        Args:
            job_id: The ID of the job.

        Returns:
            The result dictionary if the job is completed, otherwise None.
        """
        return self._completed_jobs.get(job_id)

# Example Usage:
if __name__ == "__main__":
    queue = JobQueue()

    # 1. Add jobs
    print("Adding jobs to the queue...")
    job_id1 = queue.add_job("job1", {"task": "process_data", "input": 10})
    job_id2 = queue.add_job("job2", {"task": "cleanup", "path": "/tmp"})
    job_id3 = queue.add_job("job3", {"task": "process_data", "input": 20})

    print(f"Queued jobs: {job_id1}, {job_id2}, {job_id3}\n")

    # 2. Process jobs (FIFO order: job1, job2, job3)
    print("--- Starting processing ---")
    
    # Process job 1
    queue.process_next_job()
    
    # Process job 2
    queue.process_next_job()

    # Process job 3
    queue.process_next_job()
    
    print("\n--- Processing finished ---\n")

    # 3. Get results
    
    # Check job 1 result
    result1 = queue.get_result(job_id1)
    print(f"Result for {job_id1}: {result1}")

    # Check job 2 result
    result2 = queue.get_result(job_id2)
    print(f"Result for {job_id2}: {result2}")
    
    # Check job 3 result
    result3 = queue.get_result(job_id3)
    print(f"Result for {job_id3}: {result3}")

    # Check a non-existent job
    result_none = queue.get_result("job_unknown")
    print(f"Result for job_unknown: {result_none}")