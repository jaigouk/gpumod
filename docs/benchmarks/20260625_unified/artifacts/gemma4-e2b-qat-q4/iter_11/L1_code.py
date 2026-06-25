class JobQueue:
    def __init__(self):
        # Stores job details: [{'id': job_id, 'data': job_data, 'completed': False}]
        self.queue = []

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Adds a job to the queue.
        Returns the job_id.
        """
        job = {
            "id": job_id,
            "data": data,
            "completed": False
        }
        self.queue.append(job)
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        """
        Retrieves the data of a completed job by ID.
        Returns the data if completed, otherwise None.
        """
        job = next((job for job in self.queue if job["id"] == job_id), None)

        if job and job["completed"]:
            return job["data"]

        return None

    # Helper method to simulate processing and mark jobs as complete,
    # necessary for get_result to return data based on the requirements.
    # Although not explicitly required by the method list, 
    # it allows the usage example to be functional.
    def _process_job(self, job_id: str) -> bool:
        """
        Simulates processing the job and marks it as complete.
        Returns True if processed, False if job not found.
        """
        job_index = -1
        for i, job in enumerate(self.queue):
            if job["id"] == job_id:
                job_index = i
                break

        if job_index != -1 and not self.queue[job_index]["completed"]:
            self.queue[job_index]["completed"] = True
            return True
        return False

if __name__ == '__main__':
    queue = JobQueue()

    job_id = queue.add_job("job1", {"task": "process_data"})
    print(f"Added job with ID: {job_id}")

    # Simulate processing the job
    queue._process_job(job_id)
    print(f"Job {job_id} processed.")

    # Check result
    result = queue.get_result("job1")
    print(f"Result for job1: {result}") # Expected: {'task': 'process_data'}

    # Try getting a result for a job that doesn't exist or isn't done
    result_missing = queue.get_result("job_nonexistent")
    print(f"Result for nonexistent job: {result_missing}") # Expected: None