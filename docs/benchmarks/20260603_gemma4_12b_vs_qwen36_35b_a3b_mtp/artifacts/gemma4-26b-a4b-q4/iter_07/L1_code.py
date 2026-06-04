from collections import deque
    from typing import Dict, Any, Optional

    class JobQueue:
        def __init__(self):
            self._queue = deque()
            self._jobs = {}  # Stores job info: {'data': dict, 'status': str}
            self._results = {} # Stores results: {job_id: dict}

        def add_job(self, job_id: str, data: dict) -> str:
            self._jobs[job_id] = {"data": data, "status": "queued"}
            self._queue.append(job_id)
            return job_id

        def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
            return self._results.get(job_id)

        def process_next_job(self):
            if not self._queue:
                return
            
            job_id = self._queue.popleft()
            job = self._jobs.get(job_id)
            
            if job and job["status"] == "queued":
                # Simulate processing
                job["status"] = "processing"
                # For this implementation, we'll just "echo" the data back as a result
                result = {"status": "success", "processed_data": job["data"]}
                self._results[job_id] = result
                job["status"] = "completed"