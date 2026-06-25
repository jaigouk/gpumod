import collections

class JobQueue:
    def __init__(self):
        # Queue for holding pending jobs (job_id, data)
        self.queue = collections.deque()
        # Dictionary to store results of completed jobs
        self.job_results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Add a job to the queue."""
        self.queue.append((job_id, data))
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        """
        Get the result of a job. Processes jobs in FIFO order if they are pending.
        """
        if job_id in self.job_results:
            return self.job_results[job_id]

        # If not found in results, check if the job is pending
        if self.queue:
            # Get the next job in FIFO order
            job_id, data = self.queue.popleft()

            # --- Simulated Job Processing ---
            # In a real scenario, this is where the actual work happens.
            # For this basic implementation, we simulate the result calculation.
            result = {"status": "completed", "processed_data": data.get("task")}

            # Store the result and return it
            self.job_results[job_id] = result
            return result

        return None

if __name__ == "__main__":
    queue = JobQueue()

    # Add Job 1
    job_id_1 = queue.add_job("job1", {"task": "process_data", "input_size": 100})
    print(f"Added job: {job_id_1}")

    # Add Job 2
    job_id_2 = queue.add_job("job2", {"task": "clean_data", "input_size": 50})
    print(f"Added job: {job_id_2}")

    # Retrieve Job 1 result (FIFO)
    print(f"\nRetrieving result for {job_id_1}:")
    result_1 = queue.get_result(job_id_1)
    print(result_1)

    # Retrieve Job 2 result (FIFO)
    print(f"\nRetrieving result for {job_id_2}:")
    result_2 = queue.get_result(job_id_2)
    print(result_2)

    # Retrieve non-existent job
    print(f"\nRetrieving result for job3 (non-existent):")
    result_3 = queue.get_result("job3")
    print(result_3)