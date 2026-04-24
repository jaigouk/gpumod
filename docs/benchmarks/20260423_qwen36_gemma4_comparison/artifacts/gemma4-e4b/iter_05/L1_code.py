from collections import deque
from typing import Dict, Any, Optional

class JobQueue:
    """
    A basic in-memory job queue implementation.
    """
    def __init__(self):
        # Queue of job IDs waiting to be processed (FIFO)
        self.pending_jobs: deque[str] = deque()
        # Dictionary storing all job metadata (input data, status, result)
        self.job_data: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Adds a job to the queue.

        Args:
            job_id: Unique identifier for the job.
            data: The data payload for the job.

        Returns:
            The job_id.
        """
        if job_id in self.job_data:
            raise ValueError(f"Job ID '{job_id}' already exists.")

        self.job_data[job_id] = {
            "status": "PENDING",
            "data": data,
            "result": None
        }
        self.pending_jobs.append(job_id)
        return job_id

    def process_next_job(self) -> bool:
        """
        Simulates a worker picking up and completing the next job in the queue.
        Returns True if a job was processed, False otherwise.
        """
        if not self.pending_jobs:
            return False

        job_id = self.pending_jobs.popleft()
        job = self.job_data[job_id]

        if job["status"] != "PENDING":
            # Should not happen if logic is correct, but safety check
            return False

        # --- SIMULATION OF JOB WORK ---
        # In a real system, this is where the worker thread/process would execute the task.
        input_data = job["data"]
        
        # Simple simulation: return a result based on the input
        task = input_data.get("task", "unknown")
        result_value = f"Processed successfully. Task '{task}' completed."

        # Update job status and result
        job["status"] = "COMPLETED"
        job["result"] = {"status": "SUCCESS", "output": result_value}
        # ------------------------------
        
        return True

    def get_result(self, job_id: str) -> Optional[dict]:
        """
        Retrieves the result of a completed job.

        Args:
            job_id: The ID of the job.

        Returns:
            The result dictionary if the job is completed, otherwise None.
        """
        job = self.job_data.get(job_id)
        if job and job["status"] == "COMPLETED":
            return job["result"]
        
        return None

# Example Usage:
if __name__ == '__main__':
    queue = JobQueue()

    # 1. Add jobs
    job1_id = queue.add_job("job1", {"task": "process_data", "input": 10})
    job2_id = queue.add_job("job2", {"task": "send_email", "recipient": "test@example.com"})
    job3_id = queue.add_job("job3", {"task": "cleanup", "files": 50})

    print(f"Added job 1: {job1_id}")
    print(f"Added job 2: {job2_id}")
    print("-" * 20)

    # 2. Check results before processing (should be None)
    print(f"Result for job1 before processing: {queue.get_result(job1_id)}")

    # 3. Process jobs (FIFO order: job1 -> job2 -> job3)
    print("\n--- Processing Jobs ---")
    queue.process_next_job() # Processes job1
    print("Processed job 1.")
    
    queue.process_next_job() # Processes job2
    print("Processed job 2.")
    
    queue.process_next_job() # Processes job3
    print("Processed job 3.")

    # 4. Check results after processing
    print("-" * 20)
    
    # Job 1 result
    result1 = queue.get_result(job1_id)
    print(f"Result for job1: {result1}") 

    # Job 2 result
    result2 = queue.get_result(job2_id)
    print(f"Result for job2: {result2}") 

    # Check a job that was never added
    result_missing = queue.get_result("job_missing")
    print(f"Result for missing job: {result_missing}")