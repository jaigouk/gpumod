import time

class JobQueue:
    def __init__(self):
        # Stores jobs: {job_id: {"data": dict, "attempts": int}}
        self.jobs = {}

    def add_job(self, job_id: str, data: dict):
        self.jobs[job_id] = {
            "data": data,
            "attempts": 0
        }

    def process_job(self, job_id: str, processor: callable) -> bool:
        if job_id not in self.jobs:
            return False

        job_state = self.jobs[job_id]
        max_attempts = 4
        current_attempt = job_state["attempts"]

        if current_attempt >= max_attempts:
            return False

        job_state["attempts"] += 1

        try:
            result = processor(job_state["data"])
            # If successful, reset attempts and return True
            job_state["attempts"] = 0
            return True

        except Exception:
            # Calculate exponential backoff delay (1s, 2s, 4s)
            # current_attempt is 1-indexed for calculation
            # If attempt 1 failed -> next delay is 1
            # If attempt 2 failed -> next delay is 2
            # If attempt 3 failed -> next delay is 4

            if current_attempt < max_attempts:
                delay = 2**(current_attempt - 1)
                # Record the delay data (as requested, we don't sleep)
                # In a real scenario, we would print or log this:
                print(f"Job {job_id} failed (Attempt {current_attempt}). Recording delay of {delay}s.")

            # If we failed and have attempts left, keep retrying
            # If this was the final attempt, we fall through to return False
            pass 

        return False

if __name__ == '__main__':
    # --- Example Usage ---

    queue = JobQueue()

    # 1. Define a failing processor
    failure_count = 0
    def flaky_processor(data):
        nonlocal failure_count
        print(f"\n--- Attempt processing job: {data['id']} ---")

        if failure_count < 2:
            failure_count += 1
            raise ValueError("Transient failure")

        print(f"Successfully processed job: {data['id']}")
        return True

    # 2. Add a job
    job_id = "job_101"
    job_data = {"id": job_id, "payload": "test_data"}
    queue.add_job(job_id, job_data)

    print(f"Starting processing for job: {job_id}")
    success = queue.process_job(job_id, flaky_processor)
    print(f"\nFinal Result for {job_id}: {'Success' if success else 'Failure'}")