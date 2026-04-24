import collections
from typing import Dict, Any, Optional

class JobQueue:
    """
    A basic FIFO job queue implementation.
    """
    def __init__(self):
        # Queue of job IDs waiting to be processed (FIFO)
        self._pending_jobs: collections.deque[str] = collections.deque()
        # Storage for completed job results
        self._job_results: Dict[str, Dict[str, Any]] = {}
        # Storage for initial job data (useful for processing)
        self._job_data: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> str:
        """
        Adds a job to the queue.

        Args:
            job_id: Unique identifier for the job.
            data: The data required for the job execution.

        Returns:
            The job_id.
        """
        if job_id in self._job_results or job_id in self._pending_jobs:
            raise ValueError(f"Job ID '{job_id}' already exists in the queue or results.")

        self._pending_jobs.append(job_id)
        self._job_data[job_id] = data
        return job_id

    def process_next_job(self) -> bool:
        """
        Simulates the processing of the next job in the queue.

        Returns:
            True if a job was processed, False if the queue was empty.
        """
        if not self._pending_jobs:
            return False

        job_id = self._pending_jobs.popleft()
        job_data = self._job_data.pop(job_id)

        # --- Simulation of Job Execution ---
        try:
            # Simulate work based on the job data
            task = job_data.get("task", "default_task")
            
            if task == "process_data":
                result = {"status": "completed", "processed_data": f"Result of {job_id} processed successfully."}
            elif task == "fail":
                result = {"status": "error", "message": "Simulated failure."}
            else:
                result = {"status": "completed", "message": f"Task '{task}' finished."}
            
            # Store the result
            self._job_results[job_id] = result
        
        except Exception as e:
            self._job_results[job_id] = {"status": "error", "message": str(e)}
        # -----------------------------------
        
        return True

    def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the result of a completed job.

        Args:
            job_id: The ID of the job.

        Returns:
            The result dictionary if the job is completed, otherwise None.
        """
        return self._job_results.get(job_id)

# Example Usage:
if __name__ == '__main__':
    queue = JobQueue()

    # 1. Add jobs
    job_id1 = queue.add_job("job1", {"task": "process_data", "input": 10})
    job_id2 = queue.add_job("job2", {"task": "process_data", "input": 20})
    job_id3 = queue.add_job("job3", {"task": "fail", "input": 30})

    print(f"Added jobs: {job_id1}, {job_id2}, {job_id3}")
    print("-" * 30)

    # 2. Process the first job (job1)
    print("Processing job 1...")
    queue.process_next_job()
    
    # Try to get result before processing job 2
    result1_pre = queue.get_result("job2")
    print(f"Result for job2 before processing: {result1_pre}")

    # 3. Process the second job (job2)
    print("Processing job 2...")
    queue.process_next_job()

    # 4. Check result for job1
    result1 = queue.get_result("job1")
    print(f"Result for job1: {result1}")
    
    # 5. Process the third job (job3 - failure simulation)
    print("Processing job 3...")
    queue.process_next_job()

    # 6. Check result for job3
    result3 = queue.get_result("job3")
    print(f"Result for job3: {result3}")
    
    # 7. Check result for job2
    result2 = queue.get_result("job2")
    print(f"Result for job2: {result2}")