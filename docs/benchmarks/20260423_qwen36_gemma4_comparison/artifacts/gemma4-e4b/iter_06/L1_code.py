import uuid
from collections import deque
from typing import Dict, Any, Optional

class JobQueue:
    """
    A basic in-memory job queue implementation.
    """
    def __init__(self):
        # Stores jobs waiting to be processed (FIFO order)
        # Structure: deque[job_id]
        self._pending_jobs: deque[str] = deque()
        
        # Stores job data (pending or completed)
        # Structure: dict[job_id] -> {'data': dict, 'status': str}
        self._jobs: Dict[str, Dict[str, Any]] = {}
        
        # Stores results for completed jobs
        # Structure: dict[job_id] -> dict (result data)
        self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Adds a job to the queue.

        Args:
            job_id: Unique identifier for the job.
            data: The data payload for the job.

        Returns:
            The job_id.
        """
        if job_id in self._jobs:
            raise ValueError(f"Job ID '{job_id}' already exists.")

        self._jobs[job_id] = {'data': data, 'status': 'PENDING'}
        self._pending_jobs.append(job_id)
        return job_id

    def get_result(self, job_id: str) -> Optional[dict]:
        """
        Retrieves the result of a completed job.

        Args:
            job_id: The identifier of the job.

        Returns:
            The result dictionary if the job is completed, otherwise None.
        """
        if job_id in self._results:
            return self._results[job_id]
        return None

    def process_next_job(self) -> Optional[str]:
        """
        Simulates a worker picking up and processing the next job in FIFO order.
        (This method is necessary to transition the job state and generate results.)

        Returns:
            The job_id that was processed, or None if the queue is empty.
        """
        if not self._pending_jobs:
            return None

        job_id = self._pending_jobs.popleft()
        
        job_info = self._jobs.get(job_id)
        if not job_info or job_info['status'] != 'PENDING':
            # Should not happen if queue is managed correctly
            return None

        # --- Simulation of Job Processing ---
        # In a real system, this is where the heavy lifting happens.
        data = job_info['data']
        
        # Example processing: simply return the input data plus a processed status
        try:
            # Simulate processing time/logic
            result = {"status": "COMPLETED", "input_data": data, "processed_by": "QueueWorker"}
            
            # Update state
            self._jobs[job_id]['status'] = 'COMPLETED'
            self._results[job_id] = result
            return job_id
        except Exception as e:
            # Handle failure
            self._jobs[job_id]['status'] = 'FAILED'
            print(f"Job {job_id} failed: {e}")
            return job_id


if __name__ == '__main__':
    queue = JobQueue()

    # 1. Add jobs
    job_id1 = queue.add_job("job1", {"task": "process_data", "value": 10})
    job_id2 = queue.add_job("job2", {"task": "send_email", "recipient": "test@example.com"})
    job_id3 = queue.add_job("job3", {"task": "cleanup", "files": 5})

    print(f"Added jobs: {job_id1}, {job_id2}, {job_id3}")

    # 2. Check results before processing (should be None)
    print("-" * 20)
    print(f"Result for job1 before processing: {queue.get_result(job_id1)}")

    # 3. Process jobs (FIFO order: job1, job2, job3)
    print("\n--- Processing Jobs ---")
    
    # Process job 1
    processed_id1 = queue.process_next_job()
    print(f"Processed job: {processed_id1}")

    # Process job 2
    processed_id2 = queue.process_next_job()
    print(f"Processed job: {processed_id2}")

    # Process job 3
    processed_id3 = queue.process_next_job()
    print(f"Processed job: {processed_id3}")

    # 4. Check results after processing
    print("-" * 20)
    
    # job1 should have a result
    result1 = queue.get_result(job_id1)
    print(f"Result for job1: {result1}")

    # job2 should have a result
    result2 = queue.get_result(job_id2)
    print(f"Result for job2: {result2}")

    # Try to get a result for a non-existent job
    result_none = queue.get_result("job_unknown")
    print(f"Result for unknown job: {result_none}")