from collections import deque
from typing import dict, Optional

class JobQueue:
    def __init__(self):
        # Queue for pending jobs (stores tuples of (job_id, data))
        self.pending_jobs = deque()
        # Storage for completed job results (job_id -> result dict)
        self.results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Add a job to the queue.
        Returns the unique job_id.
        """
        self.pending_jobs.append((job_id, data))
        return job_id

    def get_result(self, job_id: str) -> Optional[dict]:
        """
        Retrieve the result of a job. Processes the job if it is currently pending.
        """
        if job_id in self.results:
            return self.results[job_id]

        if not self.pending_jobs:
            return None

        # Check if the job is waiting in the queue
        found = False
        for index, (current_job_id, _) in enumerate(self.pending_jobs):
            if current_job_id == job_id:
                # Pop the job from the left (FIFO)
                job_id_popped, data_popped = self.pending_jobs.popleft()
                
                # Simulate processing and generate result
                result = {
                    "job_id": job_id_popped,
                    "status": "completed",
                    "data": data_popped,
                    "processed_at": "timestamp" 
                }
                
                # Store the result
                self.results[job_id_popped] = result
                found = True
                break
        
        return None if not found else None # Return None if job ID was never found in the pending queue

if __name__ == "__main__":
    # Example Usage
    queue = JobQueue()
    
    # Add jobs
    job_id_1 = queue.add_job("job1", {"task": "process_data"})
    job_id_2 = queue.add_job("job2", {"task": "calculate_stats"})
    
    print(f"Added jobs with IDs: {job_id_1}, {job_id_2}")

    # Retrieve result for job 1 (it gets processed first)
    result_1 = queue.get_result(job_id_1)
    print("\nResult for job1:")
    print(result_1)

    # Retrieve result for job 2
    result_2 = queue.get_result(job_id_2)
    print("\nResult for job2:")
    print(result_2)
    
    # Attempt to get result for a job that doesn't exist
    result_3 = queue.get_result("job3")
    print("\nResult for job3:")
    print(result_3)
    
    # Attempt to get result for job 1 again (already processed)
    result_1_again = queue.get_result(job_id_1)
    print("\nResult for job1 (again):")
    print(result_1_again)