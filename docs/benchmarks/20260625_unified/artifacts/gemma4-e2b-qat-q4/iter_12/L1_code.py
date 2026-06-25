from collections import deque

class JobQueue:
    def __init__(self):
        # Stores jobs in FIFO order
        self.queue = deque()
        # Stores job_id -> job_data mapping
        self.job_store = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Add a job to the queue.
        """
        if job_id in self.job_store:
            raise ValueError(f"Job ID '{job_id}' already exists")

        self.queue.append(job_id)
        self.job_store[job_id] = data
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        """
        Get the result (data) of a completed job.
        In this basic implementation, 'data' is returned upon retrieval.
        """
        if job_id in self.job_store:
            # Remove the job from the store if requested (simulating processing/completion)
            result = self.job_store.pop(job_id)
            return result
        return None