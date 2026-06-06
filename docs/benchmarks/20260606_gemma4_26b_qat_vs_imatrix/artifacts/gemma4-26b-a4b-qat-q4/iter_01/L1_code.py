from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Adds a job to the queue."""
        self._queue.append((job_id, data))
        return job_id

    def process_next_job(self, result: dict):
        """
        Simulates processing the next job in the queue (FIFO).
        Moves the job from the queue to the results storage.
        """
        if self._queue:
            job_id, _ = self._queue.popleft()
            self._results[job_id] = result

    def get_result(self, job_id: str) -> dict | None:
        """Returns the result of a completed job, or None if not found."""
        return self._results.get(job_id)

# Example usage:
if __name__ == "__main__":
    queue = JobQueue()

    # Add jobs
    id1 = queue.add_job("job1", {"task": "process_data"})
    id2 = queue.add_job("job2", {"task": "send_email"})

    # Process jobs (FIFO)
    queue.process_next_job({"status": "success", "data": [1, 2, 3]}) # Processes job1
    queue.process_next_job({"status": "error", "message": "timeout"}) # Processes job2

    # Retrieve results
    print(queue.get_result("job1")) # {'status': 'success', 'data': [1, 2, 3]}
    print(queue.get_result("job2")) # {'status': 'error', 'message': 'timeout'}
    print(queue.get_result("job3")) # None