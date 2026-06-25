from collections import deque

class JobQueue:
    """
    A basic job queue implementation that processes jobs in FIFO order.
    """
    def __init__(self):
        # Queue stores job_id's in FIFO order
        self.queue = deque()
        # Stores all job details: job_id -> {data: dict, result: dict | None}
        self.job_details = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Add a job to the queue and store its details.
        Returns the job_id.
        """
        if job_id in self.job_details:
            raise ValueError(f"Job ID '{job_id}' already exists.")

        self.queue.append(job_id)
        self.job_details[job_id] = {"data": data, "result": None}
        return job_id

    def _process_next_job(self):
        """
        Internal method to simulate processing the next job in the queue (FIFO).
        If a job exists in the queue, it moves it to completion.
        """
        if not self.queue:
            return None

        job_id = self.queue.popleft()
        job_info = self.job_details[job_id]

        # Simulation: Perform processing and set a result
        # In a real application, this is where external computation would happen.
        # For this basic implementation, we just generate a placeholder result.
        job_info["result"] = {"status": "completed", "processed_data": job_info["data"]}

        return job_id

    def get_result(self, job_id: str) -> dict | None:
        """
        Get the result of a completed job.

        Note: For this basic implementation, we simulate processing the job 
        internally when get_result is called if the job is pending.
        """
        # If the job is not tracked, return None
        if job_id not in self.job_details:
            return None

        # Simple simulation logic: if job is pending, process it to get a result.
        # This mimics a system where retrieving a job triggers its completion.
        if self.job_details[job_id]["result"] is None and job_id in self.queue:
            self._process_next_job()
            # Since we just processed it, the result is now available
            return self.job_details[job_id]["result"]

        # Return the stored result if it exists (either from a previous manual completion 
        # or from the internal processing above).
        return self.job_details[job_id].get("result")