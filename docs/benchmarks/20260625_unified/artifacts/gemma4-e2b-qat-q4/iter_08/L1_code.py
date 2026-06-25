from collections import deque

class JobQueue:
    """
    A basic implementation of a First-In-First-Out (FIFO) job queue.
    """
    def __init__(self):
        # The internal queue for pending jobs (FIFO)
        self.queue = deque()
        # Storage for completed job results: {job_id: result_data}
        self.results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Adds a job to the queue.

        Args:
            job_id: A unique identifier for the job.
            data: The payload associated with the job.

        Returns:
            The job_id assigned to the job.
        """
        job = {"id": job_id, "data": data}
        self.queue.append(job)
        return job_id

    def _process_job(self, job_id: str) -> dict:
        """
        Simulates the execution of a job and returns a result.
        (A simple placeholder for actual processing logic.)
        """
        if job_id not in self.results:
            job_data = next((item['data'] for item in self.queue if item['id'] == job_id), None)
            if job_data:
                # Simulate processing time/logic
                result = {
                    "status": "completed",
                    "input_data": job_data,
                    "processed_at": "now"
                }
                self.results[job_id] = result

                # Remove the job from the queue
                self.queue.remove(next(item for item in self.queue if item['id'] == job_id))
                return result

        return None

    def get_result(self, job_id: str) -> dict | None:
        """
        Retrieves the result of a completed job.
        If the job is not completed, this implementation simulates processing it.

        Args:
            job_id: The unique identifier of the job.

        Returns:
            The result dictionary if the job is completed, otherwise None.
        """
        # In this basic implementation, we process jobs as soon as get_result is called
        # if the job is still pending in the queue.
        return self._process_job(job_id)