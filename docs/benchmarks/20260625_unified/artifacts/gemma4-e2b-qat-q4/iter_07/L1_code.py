from collections import deque
from typing import Dict, Any, Optional

class JobQueue:
    def __init__(self):
        # Stores jobs as (job_id, data) tuples to maintain the order and hold job info
        self._queue = deque()
        # Stores completed results: {job_id: result_data}
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Adds a job to the queue. Jobs are added FIFO.
        Returns the unique job_id.
        """
        self._queue.append((job_id, data))
        return job_id

    def get_result(self, job_id: str) -> Optional[dict]:
        """
        Retrieves the result of a completed job.
        Processes jobs in FIFO order upon retrieval.
        Returns the result data if completed, otherwise None.
        """

        # Check if the job is already completed and return its result
        if job_id in self._results:
            return self._results[job_id]

        # Process the next job in the queue (FIFO)
        if self._queue:
            job_id, data = self._queue.popleft()

            # Simulate job processing: store the data as the result
            self._results[job_id] = data
            return data

        # If the queue is empty and job is not found
        return None

if __name__ == '__main__':
    queue = JobQueue()

    # 1. Add first job
    job_id_1 = queue.add_job("job1", {"task": "process_data", "input": "source_a"})
    print(f"Job added: {job_id_1}")

    # 2. Add second job
    job_id_2 = queue.add_job("job2", {"task": "clean_data", "input": "source_b"})
    print(f"Job added: {job_id_2}")

    # 3. Try to get a completed job (returns job1)
    print(f"\nRetrieving result for job1: {queue.get_result('job1')}")

    # 4. Try to get a second completed job (returns job2, FIFO)
    print(f"Retrieving result for job2: {queue.get_result('job2')}")

    # 5. Try to get a job that does not exist
    print(f"Retrieving result for job_nonexistent: {queue.get_result('job_nonexistent')}")