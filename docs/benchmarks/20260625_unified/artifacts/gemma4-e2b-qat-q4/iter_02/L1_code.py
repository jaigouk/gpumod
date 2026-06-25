class JobQueue:
    """
    A basic implementation of a First-In-First-Out (FIFO) job queue.
    """
    def __init__(self):
        # Stores jobs waiting to be processed (FIFO structure)
        self._queue = []
        # Stores completed results: {job_id: result_dict}
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Adds a job to the queue and immediately processes it (FIFO).

        Args:
            job_id: Unique identifier for the job.
            data: Dictionary containing the job payload.

        Returns:
            The job_id assigned to the job.
        """
        # 1. Store the job payload in the queue
        self._queue.append({"id": job_id, "data": data})

        # 2. Simulate processing the job
        # In a real system, this would trigger an external worker.
        print(f"Processing job: {job_id}...")

        # Simulate a result based on the input data
        simulated_result = {"status": "completed", "input": data, "processed_by": "worker_1"}

        # 3. Store the result
        self._results[job_id] = simulated_result

        return job_id

    def get_result(self, job_id: str) -> dict | None:
        """
        Retrieves the result of a completed job.

        Args:
            job_id: The unique identifier of the job.

        Returns:
            The result dictionary if the job exists, otherwise None.
        """
        return self._results.get(job_id)

if __name__ == '__main__':
    # Example usage
    queue = JobQueue()

    # Add first job
    job_id_1 = queue.add_job("job1", {"task": "process_data"})
    print(f"Job added with ID: {job_id_1}")

    # Add second job
    job_id_2 = queue.add_job("job2", {"task": "generate_report"})
    print(f"Job added with ID: {job_id_2}")

    print("\n--- Checking Results ---")

    # Get result for job1
    result1 = queue.get_result(job_id_1)
    print(f"Result for {job_id_1}: {result1}")

    # Get result for job2
    result2 = queue.get_result("job2")
    print(f"Result for job2: {result2}")

    # Check non-existent job
    result_nonexistent = queue.get_result("job3")
    print(f"Result for job3 (non-existent): {result_nonexistent}")