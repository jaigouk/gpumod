from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._jobs = {}
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._jobs[job_id] = data
        self._queue.append(job_id)
        return job_id

    def process_next_job(self, processing_func):
        """
        Processes the next job in the FIFO queue using the provided function.
        """
        if not self._queue:
            return None

        job_id = self._queue.popleft()
        data = self._jobs.pop(job_id)

        # Execute the task and store the result
        result = processing_func(data)
        self._results[job_id] = result
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id)

# Example usage to demonstrate functionality:
if __name__ == "__main__":
    queue = JobQueue()

    # 1. Add a job
    job_id = queue.add_job("job1", {"task": "process_data", "value": 10})

    # 2. Define a dummy processing function
    def dummy_worker(data):
        return {"status": "success", "output": data["value"] * 2}

    # 3. Process the job
    queue.process_next_job(dummy_worker)

    # 4. Get the result
    result = queue.get_result("job1")
    print(result)  # {'status': 'success', 'output': 20}