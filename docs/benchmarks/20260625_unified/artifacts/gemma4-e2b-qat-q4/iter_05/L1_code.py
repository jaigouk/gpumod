import time
from collections import deque

class JobQueue:
    """
    A basic implementation of a FIFO job queue.
    """
    def __init__(self):
        # Stores (job_id, data) tuples
        self.queue = deque()
        # Stores the actual job data mapping job_id to data
        self.jobs_data = {}
        # Stores job results mapping job_id to result
        self.results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Add a job to the queue.

        Args:
            job_id: Unique identifier for the job.
            data: Dictionary containing job parameters.

        Returns:
            The job_id returned upon successful addition.
        """
        if job_id in self.jobs_data:
            raise ValueError(f"Job ID '{job_id}' already exists.")

        self.jobs_data[job_id] = data
        self.queue.append(job_id)
        return job_id

    def process_next(self) -> str | None:
        """
        Processes the next job in the queue (FIFO).
        If a job is processed, its result is calculated and stored.

        Returns:
            The job_id of the processed job, or None if the queue is empty.
        """
        if not self.queue:
            return None

        job_id = self.queue.popleft()

        data = self.jobs_data.get(job_id)
        if not data:
            # Should not happen if logic is sound
            return None

        print(f"Processing job: {job_id} with data: {data}")

        # --- Simulation of Job Processing ---
        time.sleep(0.1)  # Simulate work

        # Simulate result generation based on task
        result = {"status": "completed", "processed_at": time.time()}
        if data.get("task") == "process_data":
            result["output"] = f"Data processed successfully: {data['input_count'] * 10}"
        else:
            result["output"] = f"Task {data['task']} finished."

        self.results[job_id] = result
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        """
        Get the result of a completed job.

        Note: In this implementation, a job is considered 'completed' 
        only after it has been processed by process_next().

        Args:
            job_id: The ID of the job to retrieve the result for.

        Returns:
            The job result dictionary, or None if the job ID does not exist 
            or the job is not yet completed.
        """
        # Check if the job has been processed and has a result
        if job_id in self.results:
            return self.results[job_id]

        return None

# Example Usage:
if __name__ == "__main__":
    queue = JobQueue()

    # Add jobs
    queue.add_job("job1", {"task": "process_data", "input_count": 5})
    queue.add_job("job2", {"task": "calculate_sum", "val_a": 10, "val_b": 20})
    job_id_3 = queue.add_job("job3", {"task": "process_data", "input_count": 2})

    print("--- Queue initialized ---")

    # Get result before processing (should be None)
    print(f"\nAttempting to get result for job1 before processing: {queue.get_result('job1')}")

    print("\n--- Processing jobs (FIFO) ---")

    # Process job1
    processed_id_1 = queue.process_next()
    if processed_id_1:
        print(f"Job {processed_id_1} successfully processed.")
        result1 = queue.get_result(processed_id_1)
        print(f"Result for job1: {result1}")

    # Process job2
    processed_id_2 = queue.process_next()
    if processed_id_2:
        print(f"Job {processed_id_2} successfully processed.")
        result2 = queue.get_result(processed_id_2)
        print(f"Result for job2: {result2}")

    # Process job3
    processed_id_3 = queue.process_next()
    if processed_id_3:
        print(f"Job {processed_id_3} successfully processed.")
        result3 = queue.get_result(processed_id_3)
        print(f"Result for job3: {result3}")

    # Attempt to get result for an unfinished job (if we added a job here)
    print(f"\nAttempting to get result for job1 again: {queue.get_result('job1')}")