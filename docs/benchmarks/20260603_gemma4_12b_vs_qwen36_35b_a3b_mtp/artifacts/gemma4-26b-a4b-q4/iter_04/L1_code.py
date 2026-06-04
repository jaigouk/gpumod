from collections import deque

    class JobQueue:
        def __init__(self):
            self.queue = deque()
            self.results = {}

        def add_job(self, job_id: str, data: dict) -> str:
            self.queue.append((job_id, data))
            return job_id

        def process_next(self): # Necessary to satisfy FIFO and "processing" requirement
            if not self.queue:
                return None
            job_id, data = self.queue.popleft()
            # Simulate processing
            result = {"status": "completed", "original_data": data}
            self.results[job_id] = result
            return result

        def get_result(self, job_id: str) -> dict | None:
            return self.results.get(job_id)