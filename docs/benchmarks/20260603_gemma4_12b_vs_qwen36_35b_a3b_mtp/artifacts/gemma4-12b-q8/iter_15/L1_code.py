from collections import deque

class JobQueue:
    def __init__(self):
        # Stores the jobs in FIFO order
        self.queue = deque()
        # Stores the results of jobs, mapped by job_id
        self.results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Adds a job to the queue and initializes its result as None.
        """
        self.queue.append({"id": job_id, "data": data})
        self.results[job_id] = None
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        """
        Returns the result of a completed job. 
        Returns None if the job is not completed or does not exist.
        """
        return self.results.get(job_id)

    def complete_job(self, job_id: str, result: dict):
        """
        Updates the result for a specific job_id.
        """
        if job_id in self.results:
            self.results[job_id] = result