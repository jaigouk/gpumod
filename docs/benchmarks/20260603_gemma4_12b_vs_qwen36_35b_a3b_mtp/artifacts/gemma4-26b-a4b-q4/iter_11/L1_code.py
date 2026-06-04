from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Adds a job to the FIFO queue."""
        self._queue.append((job_id, data))
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        """Returns the result of a job if it has been processed."""
        return self._results.get(job_id)

    def process_next_job(self) -> None:
        """
        Processes the next job in the queue (FIFO).
        In a real application, this might run in a separate thread or worker.
        """
        if self._queue:
            job_id, data = self._queue.popleft()
            # Simulate processing by storing a result dictionary
            self._results[job_id] = {
                "status": "completed",
                "processed_data": data
            }

# Example usage:
if __name__ == "__main__":
    queue = JobQueue()

    # Add jobs
    job_id1 = queue.add_job("job1", {"task": "process_data"})
    job_id2 = queue.add_job("job2", {"task": "upload_file"})

    # Process jobs in FIFO order
    queue.process_next_job()  # Processes job1
    queue.process_next_job()  # Processes job2

    # Retrieve results
    print(f"Result for {job_id1}: {queue.get_result(job_id1)}")
    print(f"Result for {job_id2}: {queue.get_result(job_id2)}")
    print(f"Result for non-existent: {queue.get_result('job3')}")