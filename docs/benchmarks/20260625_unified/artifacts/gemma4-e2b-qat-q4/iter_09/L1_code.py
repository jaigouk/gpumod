import collections
import uuid
import random

class JobQueue:
    def __init__(self):
        # Queue stores dictionaries representing pending jobs
        # Each job object contains its own data and status
        self.queue = collections.deque()
        # Dictionary to store results of completed jobs: {job_id: result_data}
        self.completed_results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Add a job to the queue.

        Args:
            job_id: Unique identifier for the job.
            data: The input data required for the job task.

        Returns:
            The job_id assigned to the job.
        """
        job = {
            "id": job_id,
            "data": data,
            "status": "pending"
        }
        self.queue.append(job)
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        """
        Retrieve the result of a completed job.

        In this implementation, calling get_result triggers the processing of the 
        next job in the FIFO order if it is currently pending.

        Args:
            job_id: The ID of the job to retrieve the result for.

        Returns:
            The result dictionary if the job is processed and completed, 
            otherwise None.
        """
        # 1. Check if the result is already available
        if job_id in self.completed_results:
            return self.completed_results[job_id]

        # 2. Process the next pending job (FIFO logic)
        if self.queue:
            job = self.queue.popleft()
            job_id = job["id"]

            # Simulate processing the job
            try:
                # Simulate actual work based on the data
                task_data = job["data"].get("task", "unknown task")

                # Generate a mock result
                result = {
                    "job_id": job_id,
                    "status": "completed",
                    "output": f"Task '{task_data}' successfully processed."
                }

                # Store the result
                self.completed_results[job_id] = result
                return result

            except Exception as e:
                # In a real system, error handling would be more robust
                print(f"Error processing job {job_id}: {e}")
                return None

        return None

if __name__ == '__main__':
    queue = JobQueue()

    job1_id = queue.add_job("job1", {"task": "process_data"})
    job2_id = queue.add_job("job2", {"task": "generate_report"})

    print(f"Job 1 added with ID: {job1_id}")
    print(f"Job 2 added with ID: {job2_id}")

    print("\n--- Retrieving Job 1 result (Triggers processing of Job 1) ---")
    result1 = queue.get_result(job1_id)
    print(result1)

    print("\n--- Retrieving Job 2 result (Triggers processing of Job 2) ---")
    result2 = queue.get_result(job2_id)
    print(result2)

    print("\n--- Attempting to retrieve Job 1 result again (Already complete) ---")
    result1_again = queue.get_result(job1_id)
    print(result1_again)