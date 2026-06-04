from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Add a job to the queue."""
        self._queue.append((job_id, data))
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        """Get the result of a completed job."""
        return self._results.get(job_id)

    def process_next_job(self):
        """
        Simulates a worker picking up and processing the next job 
        in the queue (FIFO).
        """
        if self._queue:
            job_id, data = self._queue.popleft()
            # Simulate processing logic
            result = {
                "status": "completed",
                "processed_data": data
            }
            self._results[job_id] = result

# Example usage:
if __name__ == "__main__":
    queue = JobQueue()
    job_id = queue.add_job("job1", {"task": "process_data"})

    # Before processing
    print(f"Result for {job_id}: {queue.get_result(job_id)}") 

    # Process the job
    queue.process_next_job()

    # After processing
    result = queue.get_result(job_id)
    print(f"Result for {job_id}: {result}")