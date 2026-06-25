from collections import deque

class JobQueue:
    """
    A basic FIFO job queue implementation.
    """
    def __init__(self):
        # Queue stores jobs (job_id -> data) in FIFO order
        self.queue = deque()
        # Dictionary to store job results (job_id -> result)
        self.results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Add a job to the queue.

        Args:
            job_id: Unique identifier for the job.
            data: Data payload for the job.

        Returns:
            The job_id assigned to the job.
        """
        if job_id in self.results:
            raise ValueError(f"Job ID {job_id} already exists.")

        job_info = {
            "id": job_id,
            "data": data,
            "status": "pending"
        }

        self.queue.append(job_info)
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        """
        Retrieves and processes the next job from the queue in FIFO order,
        simulating its completion.

        Args:
            job_id: The ID of the job to retrieve and process.

        Returns:
            The result of the processed job, or None if not found.
        """
        # 1. Check if the job exists in the results
        if job_id in self.results:
            return self.results[job_id]

        # 2. Find the job in the queue
        # We iterate through the queue (FIFO) to find the job ID
        for i, job in enumerate(self.queue):
            if job["id"] == job_id:
                # Remove job from queue (FIFO)
                self.queue.pop(i)

                # Simulate processing and set the result
                job["status"] = "completed"
                # Simple mock result calculation
                result = {"processed_data": job["data"].get("task", "default_task"), "status": "success"}
                self.results[job_id] = result
                return result

        # Job not found
        return None

# Example Usage:
if __name__ == "__main__":
    queue = JobQueue()

    print("--- Adding jobs ---")
    job1 = queue.add_job("job1", {"task": "process_data"})
    job2 = queue.add_job("job2", {"task": "generate_report"})
    job3 = queue.add_job("job3", {"task": "clean_database"})

    print(f"Added Job 1: {job1}")
    print(f"Added Job 2: {job2}")
    print(f"Added Job 3: {job3}\n")

    print("--- Retrieving Job 2 result (Should be None initially) ---")
    result_none = queue.get_result("job2")
    print(f"Job 2 result: {result_none}\n")

    print("--- Retrieving Job 1 result (Processing starts) ---")
    result1 = queue.get_result(job1)
    print(f"Job 1 result: {result1}")
    print(f"Current queue size: {len(queue.queue)}\n") # Should be 2

    print("--- Retrieving Job 3 result (Processing starts) ---")
    result3 = queue.get_result(job3)
    print(f"Job 3 result: {result3}")
    print(f"Current queue size: {len(queue.queue)}\n") # Should be 1

    print("--- Retrieving Job 2 result (Now in results dictionary) ---")
    result2 = queue.get_result(job2)
    print(f"Job 2 result: {result2}")
    print(f"Current queue size: {len(queue.queue)}\n") # Should be 0

    print("--- Retrieving non-existent job ---")
    result_none_2 = queue.get_result("non_existent")
    print(f"Non-existent job result: {result_none_2}")