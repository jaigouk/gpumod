class JobQueue:
    def __init__(self):
        # Stores pending jobs as tuples: (job_id, data)
        self._queue = []
        # Stores results for completed jobs: {job_id: data}
        self._results = {}
        # Counter to ensure unique job_ids if IDs aren't provided externally, 
        # but per requirement, job_id is passed externally.

    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append((job_id, data))
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        if job_id in self._results:
            return self._results[job_id]

        return None

    # Helper method to simulate processing to make get_result useful
    def _process_job(self, job_id: str) -> bool:
        """Simulates picking the next job from the queue and completing it."""
        if not self._queue:
            return False

        # FIFO: Get the first job
        job_id_to_process, job_data = self._queue.pop(0)

        # Simulate processing
        # In a real system, this is where the actual work happens
        print(f"Processing job: {job_id_to_process}")

        # Store the result
        self._results[job_id_to_process] = job_data
        return True

    # Optional: A method to retrieve the next job for demonstration of FIFO flow
    def process_next_job(self) -> str | None:
        """Moves the next job from the queue to the results."""
        return self._process_job(self._queue[0][0]) if self._queue else None


if __name__ == '__main__':
    # --- Example Usage ---
    queue = JobQueue()

    print("--- Adding Jobs ---")
    job_id_1 = queue.add_job("job1", {"task": "process_data", "input": "A"})
    job_id_2 = queue.add_job("job2", {"task": "cleanup", "input": "B"})

    print(f"Added job1 with ID: {job_id_1}")
    print(f"Added job2 with ID: {job_id_2}\n")

    print("--- Checking results before processing ---")
    result_1 = queue.get_result(job_id_1)
    print(f"Result for {job_id_1}: {result_1}") # Should be None

    print("\n--- Processing job1 ---")
    queue.process_next_job()

    print("\n--- Checking results after processing job1 ---")
    result_1_after = queue.get_result(job_id_1)
    print(f"Result for {job_id_1}: {result_1_after}") # Should contain data

    print("\n--- Processing job2 ---")
    queue.process_next_job()

    print("\n--- Checking results for job2 ---")
    result_2 = queue.get_result(job_id_2)
    print(f"Result for {job_id_2}: {result_2}") # Should contain data

    print("\n--- Checking result for a non-existent job ---")
    result_3 = queue.get_result("job_nonexistent")
    print(f"Result for job_nonexistent: {result_3}") # Should be None