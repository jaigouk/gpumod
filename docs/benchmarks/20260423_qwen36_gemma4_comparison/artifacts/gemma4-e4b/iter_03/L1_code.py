import uuid
from collections import deque
import time
from typing import Dict, Any, Optional

class JobQueue:
    """
    A basic in-memory job queue implementation.
    Simulates FIFO processing and result retrieval.
    """
    def __init__(self):
        # Queue for jobs waiting to be processed (FIFO)
        self._pending_jobs: deque[Dict[str, Any]] = deque()
        # Storage for job results (job_id -> result_data)
        self._job_results: Dict[str, Dict[str, Any]] = {}
        # Storage for job status (job_id -> 'PENDING' or 'COMPLETED')
        self._job_statuses: Dict[str, str] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> str:
        """
        Adds a job to the queue.
        :param job_id: Unique identifier for the job.
        :param data: The data/payload for the job task.
        :return: The job_id.
        """
        if job_id in self._job_statuses:
            raise ValueError(f"Job ID {job_id} already exists.")

        job = {
            "id": job_id,
            "data": data,
            "status": "PENDING"
        }
        self._pending_jobs.append(job)
        self._job_statuses[job_id] = "PENDING"
        return job_id

    def process_next_job(self) -> Optional[str]:
        """
        Simulates a worker picking up and processing the next job in the queue.
        If a job is processed successfully, its result is stored.
        :return: The job_id of the completed job, or None if the queue is empty.
        """
        if not self._pending_jobs:
            return None

        job = self._pending_jobs.popleft()
        job_id = job['id']
        data = job['data']

        print(f"[Worker] Starting processing job: {job_id} with data: {data}")

        # --- Simulated Processing Logic ---
        try:
            # Simulate time taken for processing
            time.sleep(0.1)

            # Dummy processing function:
            result_data = {
                "status": "SUCCESS",
                "processed_data": data.get("task", "unknown_task").upper(),
                "timestamp": time.time()
            }

            # Store the result and update status
            self._job_results[job_id] = result_data
            self._job_statuses[job_id] = "COMPLETED"
            print(f"[Worker] Finished processing job: {job_id}")
            return job_id

        except Exception as e:
            # Handle potential failures
            self._job_statuses[job_id] = "FAILED"
            print(f"[Worker] Job {job_id} failed: {e}")
            return job_id

    def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the result of a completed job.
        :param job_id: The ID of the job.
        :return: The result dictionary if completed, otherwise None.
        """
        if job_id not in self._job_statuses:
            return None

        status = self._job_statuses[job_id]
        
        if status == "COMPLETED":
            return self._job_results.get(job_id)
        
        elif status == "PENDING":
            # Job is still waiting in the queue
            return None
        
        else:
            # Job failed or status unknown
            return None


if __name__ == '__main__':
    queue = JobQueue()

    # 1. Add jobs
    job_id_1 = queue.add_job("job1", {"task": "process_data", "params": 10})
    job_id_2 = queue.add_job("job2", {"task": "cleanup_files", "params": 5})
    job_id_3 = queue.add_job("job3", {"task": "generate_report", "params": 1})

    print("-" * 30)
    print(f"Added job 1: {job_id_1}")
    print(f"Added job 2: {job_id_2}")
    print(f"Added job 3: {job_id_3}")
    print("-" * 30)

    # 2. Check results before processing (should be None)
    print("Checking results before processing:")
    print(f"Result job1: {queue.get_result(job_id_1)}")

    # 3. Process jobs sequentially (FIFO)
    print("\n--- Starting Processing ---")
    
    # Process job 1 (FIFO)
    queue.process_next_job()
    
    # Process job 2
    queue.process_next_job()
    
    # Process job 3
    queue.process_next_job()

    print("\n--- Processing Complete ---")

    # 4. Check results after processing
    print("\nChecking results after processing:")
    
    # Check job 1 result
    result_1 = queue.get_result(job_id_1)
    print(f"Result job1: {result_1}")

    # Check job 3 result
    result_3 = queue.get_result(job_id_3)
    print(f"Result job3: {result_3}")

    # Check non-existent job
    result_none = queue.get_result("non_existent_job")
    print(f"Result non_existent: {result_none}")