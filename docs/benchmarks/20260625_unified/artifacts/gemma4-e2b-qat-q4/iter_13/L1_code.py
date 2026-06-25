from collections import deque

class JobQueue:
    def __init__(self):
        # Stores jobs waiting to be processed (FIFO order)
        self.queue = deque()
        # Stores results of completed jobs {job_id: result_data}
        self.results = {}
        # Stores pending jobs {job_id: data}
        self.pending_jobs = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Add a job to the queue.
        :param job_id: Unique identifier for the job.
        :param data: The payload associated with the job.
        :return: The job_id.
        """
        if job_id in self.pending_jobs:
            raise ValueError(f"Job ID {job_id} already exists.")

        job_record = {"data": data, "status": "pending"}
        self.pending_jobs[job_id] = job_record
        self.queue.append(job_id)
        return job_id

    def _process_next_job(self):
        """Internal method to simulate processing the next job in the queue."""
        if not self.queue:
            return None

        # Get the next job ID (FIFO)
        job_id = self.queue.popleft()
        job_data = self.pending_jobs.pop(job_id)['data']

        # Simulate processing the job (e.g., take 2 seconds, or perform actual logic)
        print(f"Processing job: {job_id} with data: {job_data}")

        # In a real scenario, this is where the task execution happens.
        # For simulation, we generate a placeholder result.
        result = {"status": "completed", "processed_data": job_data}

        # Store the result
        self.results[job_id] = result

        return job_id

    def get_result(self, job_id: str) -> dict | None:
        """
        Get the result of a completed job.
        If the job is pending, returns None.
        """
        if job_id in self.results:
            return self.results[job_id]

        # Check if it's in pending list (meaning it hasn't been processed yet)
        if job_id in self.pending_jobs:
            return None

        return None

    def process_jobs_until_empty(self):
        """
        Processes all jobs currently in the queue until empty.
        """
        while self.queue:
            self._process_next_job()
        print("All jobs processed.")


if __name__ == '__main__':
    queue = JobQueue()

    # 1. Add jobs
    job1_id = queue.add_job("job1", {"task": "process_data", "input": 100})
    job2_id = queue.add_job("job2", {"task": "calculate_sum", "values": [1, 2, 3]})

    print(f"Added job {job1_id} and {job2_id}.")
    print("-" * 20)

    # 2. Attempt to get results before processing
    result = queue.get_result(job1_id)
    print(f"Result for {job1_id} before processing: {result}") # Should be None

    print("\n" + "=" * 20)

    # 3. Simulate processing the jobs
    print("Starting job processing sequence...")
    queue.process_jobs_until_empty()
    print("=" * 20)

    # 4. Get results after processing
    result1 = queue.get_result(job1_id)
    print(f"Result for {job1_id} after processing: {result1}")

    result2 = queue.get_result(job2_id)
    print(f"Result for {job2_id} after processing: {result2}")