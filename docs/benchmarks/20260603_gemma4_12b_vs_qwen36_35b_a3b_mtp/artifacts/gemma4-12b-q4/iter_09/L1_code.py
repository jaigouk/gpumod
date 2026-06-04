from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        self._queue: deque[str] = deque()
        self._jobs: Dict[str, dict] = {}
        self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Add a job to the queue.
        """
        if job_id in self._jobs or job_id in self._results:
            raise ValueError("Job ID must be unique.")
        
        self._jobs[job_id] = data
        self._queue.append(job_id)
        return job_id

    def process_next(self) -> Optional[str]:
        """
        Processes the next job in the FIFO queue.
        This method is used to transition a job from 'pending' to 'completed'.
        """
        if not self._queue:
            return None
            
        job_id = self._queue.popleft()
        job_data = self._jobs.pop(job_id)
        
        # Simulate processing logic
        result = {"status": "completed", "original_data": job_data}
        self._results[job_id] = result
        return job_id

    def get_result(self, job_id: str) -> Optional[dict]:
        """
        Get the result of a completed job.
        """
        return self._results.get(job_id)

# Example usage:
if __name__ == "__main__":
    queue = JobQueue()
    job_id = queue.add_job("job1", {"task": "process_data"})
    
    # Process the job
    queue.process_next()
    
    # Get the result
    result = queue.get_result("job1")
    print(result)