import uuid
from collections import deque

class JobQueue:
    """
    A basic FIFO job queue implementation.
    """
    def __init__(self):
        # Stores jobs waiting to be processed: deque of (job_id, data)
        self._pending_jobs = deque()
        # Stores results of completed jobs: {job_id: result_data}
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Adds a job to the queue.

        Args:
            job_id: The unique identifier for the job.
            data: The data required for the job.

        Returns:
            The job_id.
        """
        if not job_id:
            # Generate a unique ID if none is provided, though the requirement implies it is provided.
            job_id = str(uuid.uuid4())
        
        # Store the job tuple
        self._pending_jobs.append((job_id, data))
        return job_id

    def process_next_job(self) -> str | None:
        """
        Simulates a worker picking up and processing the next job in FIFO order.
        
        In a real system, this method would run the worker function.
        Here, it simulates completion and stores a dummy result.

        Returns:
            The job_id that was processed, or None if the queue is empty.
        """
        if not self._pending_jobs:
            return None

        job_id, data = self._pending_jobs.popleft()
        
        # --- Simulation of Job Processing ---
        # In a real scenario, we would execute a function here:
        # result = execute_task(data)
        
        # Dummy result simulation:
        result = {"status": "completed", "input_data": data, "processed_by": "QueueWorker"}
        
        # Store the result
        self._results[job_id] = result
        
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        """
        Retrieves the result of a completed job.

        Args:
            job_id: The identifier of the job.

        Returns:
            The result dictionary if the job is completed, otherwise None.
        """
        return self._results.get(job_id)

if __name__ == '__main__':
    # --- Example Usage ---
    queue = JobQueue()

    # 1. Add jobs
    job_id1 = queue.add_job("job1", {"task": "process_data", "payload": 10})
    job_id2 = queue.add_job("job2", {"task": "send_email", "user": "alice"})
    job_id3 = queue.add_job("job3", {"task": "cleanup", "path": "/tmp"})

    print(f"Added Job 1: {job_id1}")
    print(f"Added Job 2: {job_id2}")
    print(f"Added Job 3: {job_id3}")
    print("-" * 30)

    # 2. Process jobs (Simulation)
    print("Starting job processing...")
    
    # Process Job 1 (FIFO)
    processed_id = queue.process_next_job()
    print(f"Processed job: {processed_id}")

    # Process Job 2
    processed_id = queue.process_next_job()
    print(f"Processed job: {processed_id}")
    
    # Process Job 3
    processed_id = queue.process_next_job()
    print(f"Processed job: {processed_id}")
    
    # Try processing when empty
    processed_id = queue.process_next_job()
    print(f"Attempted processing empty queue: {processed_id}")
    print("-" * 30)


    # 3. Retrieve results
    print("Retrieving results:")
    
    # Get result for Job 1
    result1 = queue.get_result(job_id1)
    print(f"Result for Job 1: {result1}")

    # Get result for Job 2
    result2 = queue.get_result(job_id2)
    print(f"Result for Job 2: {result2}")

    # Get result for a job that doesn't exist
    result_missing = queue.get_result("nonexistent_job")
    print(f"Result for missing job: {result_missing}")