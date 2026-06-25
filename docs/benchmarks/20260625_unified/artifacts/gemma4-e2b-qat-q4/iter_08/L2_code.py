class JobQueue:
    def __init__(self):
        self.jobs = {}

    def add_job(self, job_id: str, data: dict):
        """Adds a job to the queue."""
        self.jobs[job_id] = {"data": data, "retries": 0}

    def process_job(self, job_id: str, processor: Callable) -> bool:
        """
        Processes a job with retry logic and exponential backoff simulation.
        """
        if job_id not in self.jobs:
            return False

        job_state = self.jobs[job_id]
        data = job_state["data"]
        max_attempts = 4

        for attempt in range(max_attempts):
            try:
                # 1. Call processor with data
                processor(data)
                
                # Success
                return True
            
            except Exception as e:
                # 2. On any exception, handle retry logic
                if attempt < max_attempts - 1:
                    # 3. Track retry count
                    job_state["retries"] = attempt
                    
                    # 4. Calculate exponential backoff (1, 2, 4)
                    delay = 2 ** attempt 
                    
                    # 4. Do NOT actually sleep — record the delay as data/logic simulation
                    # In a real implementation, we would sleep here. 
                    # Here, we just proceed to the next attempt.
                    pass 
                else:
                    # 5. Return False if all 4 attempts fail
                    # 5. Return False if all 4 attempts raise.
                    return False

        return False


# Example usage (Optional, for internal testing):
if __name__ == '__main__':
    q = JobQueue()

    # Test Case 1: Immediate success (Processor works first try)
    def success_processor(data):
        return True
    
    job1_id = "job1"
    q.add_job(job1_id, {"payload": "data1"})
    print(f"Processing {job1_id}: {q.process_job(job1_id, success_processor)}") 

    print("-" * 20)

    # Test Case 2: Failure requiring 3 retries (total 4 attempts)
    fail_count = 0
    def failure_processor(data):
        global fail_count
        fail_count += 1
        if fail_count <= 3:
            raise Exception("Processor failed")
        return True

    job2_id = "job2"
    q.add_job(job2_id, {"payload": "data2"})
    print(f"Processing {job2_id}: {q.process_job(job2_id, failure_processor)}")

    print(f"Job {job2_id} final retries: {q.jobs[job2_id]['retries']}")
    
    print("-" * 20)

    # Test Case 3: Total failure (5 attempts expected if 4 max attempts fail)
    fail_count = 0
    def always_fail_processor(data):
        global fail_count
        fail_count += 1
        raise Exception("Fatal failure")
    
    job3_id = "job3"
    q.add_job(job3_id, {"payload": "data3"})
    print(f"Processing {job3_id}: {q.process_job(job3_id, always_fail_processor)}")
    print(f"Job {job3_id} final retries: {q.jobs[job3_id]['retries']}")