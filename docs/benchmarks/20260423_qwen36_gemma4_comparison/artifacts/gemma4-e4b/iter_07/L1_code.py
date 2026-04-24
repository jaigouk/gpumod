import uuid
from collections import deque

class JobQueue:
    """
    A basic FIFO job queue implementation.
    """
    def __init__(self):
        # Stores jobs awaiting processing (FIFO)
        self._pending_queue = deque()
        # Stores job data and status (for tracking)
        self._job_details = {}
        # Stores the final results of completed jobs
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Adds a job to the queue.

        :param job_id: The unique identifier for the job.
        :param data: The data payload for the job.
        :return: The job_id.
        """
        if job_id in self._job_details:
            raise ValueError(f"Job ID {job_id} already exists.")

        self._job_details[job_id] = {
            "data": data,
            "status": "PENDING"
        }
        self._pending_queue.append(job_id)
        return job_id

    def process_next_job(self) -> str | None:
        """
        Simulates the processing of the next job in the queue.
        Moves a job from PENDING to COMPLETED.

        :return: The job_id that was processed, or None if the queue is empty.
        """
        if not self._pending_queue:
            return None

        job_id = self._pending_queue.popleft()
        job_info = self._job_details.get(job_id)

        if job_info and job_info["status"] == "PENDING":
            # --- Simulation of actual work ---
            # In a real system, this is where the worker executes the task.
            # We simulate a successful result based on the input data.
            result = {"status": "SUCCESS", "processed_data": job_info["data"]}
            
            self._results[job_id] = result
            job_info["status"] = "COMPLETED"
            return job_id
        
        return None

    def get_result(self, job_id: str) -> dict | None:
        """
        Retrieves the result of a completed job.

        :param job_id: The unique identifier of the job.
        :return: The result dictionary if completed, otherwise None.
        """
        if job_id in self._results:
            return self._results[job_id]
        return None

# Example Usage:
if __name__ == "__main__":
    queue = JobQueue()

    # 1. Add jobs
    job_id_1 = "job1"
    job_id_2 = "job2"
    job_id_3 = "job3"

    queue.add_job(job_id_1, {"task": "process_data", "input": 10})
    queue.add_job(job_id_2, {"task": "send_email", "user": "alice"})
    queue.add_job(job_id_3, {"task": "cleanup", "files": 5})

    print("--- Queue Setup Complete ---")
    print(f"Jobs added: {job_id_1}, {job_id_2}, {job_id_3}")

    # 2. Process jobs sequentially (FIFO)
    print("\n--- Processing Jobs ---")
    
    # Process job 1
    processed_id_1 = queue.process_next_job()
    print(f"Processed: {processed_id_1}")

    # Check result for job 1
    result_1 = queue.get_result(job_id_1)
    print(f"Result for {job_id_1}: {result_1}")

    # Process job 2
    processed_id_2 = queue.process_next_job()
    print(f"Processed: {processed_id_2}")

    # Check result for job 2
    result_2 = queue.get_result(job_id_2)
    print(f"Result for {job_id_2}: {result_2}")

    # Attempt to get result for job 3 (which hasn't been processed yet)
    result_3_unprocessed = queue.get_result(job_id_3)
    print(f"Result for {job_id_3} (before processing): {result_3_unprocessed}")

    # Process job 3
    processed_id_3 = queue.process_next_job()
    print(f"Processed: {processed_id_3}")

    # Check final result for job 3
    result_3 = queue.get_result(job_id_3)
    print(f"Result for {job_id_3} (after processing): {result_3}")