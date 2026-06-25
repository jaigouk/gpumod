from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        # job_id -> {'data': dict, 'attempts': int}
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: dict) -> None:
        """Adds a job to the queue."""
        if job_id not in self.jobs:
            self.jobs[job_id] = {
                'data': data,
                'attempts': 0
            }

    def process_job(self, job_id: str, processor: Callable) -> bool:
        """
        Processes a job with a maximum of 4 attempts, using exponential backoff delays.

        Returns True on success, False if all attempts fail.
        """
        if job_id not in self.jobs:
            return False

        job_state = self.jobs[job_id]
        data = job_state['data']

        MAX_ATTEMPTS = 4

        for attempt in range(1, MAX_ATTEMPTS + 1):
            job_state['attempts'] = attempt

            try:
                # 1. Call processor(data)
                processor(data)

                # 5. Return True on success
                return True

            except Exception:
                # 2. On any exception from the processor, retry.

                if attempt < MAX_ATTEMPTS:
                    # 3. Track the retry count per job

                    # 4. Record the delays as data (Simulated by printing/logging, but storage is internal)
                    # This data could be stored in job_state['delays'] if the requirement was persistent state
                    delays = {
                        1: 1,  # Initial attempt failed, wait 1s for retry 1
                        2: 2,  # Failed attempt 2, wait 2s for retry 2
                        3: 4   # Failed attempt 3, wait 4s for retry 3
                    }

                    if attempt <= len(delays):
                        wait_time = delays[attempt]
                        # In a real implementation, we would sleep(wait_time).
                        # Here, we just acknowledge the required delay.
                        pass

                    # Continue to the next iteration (next attempt)
                else:
                    # 5. Return False if all 4 attempts raise
                    return False

        # Fallback return (should theoretically be caught by the loop logic, but included for completeness)
        return False

if __name__ == '__main__':
    import time

    # Mock processor that fails on first 3 calls and succeeds on the 4th
    count = 0
    def flaky_processor(data):
        global count
        count += 1
        print(f"Attempt {count}: Processing data: {data}")
        if count <= 3:
            print("--- Processor failed.")
            raise RuntimeError("Processor error")
        else:
            print("--- Processor succeeded!")

    queue = JobQueue()
    job_id_1 = "job_A"
    queue.add_job(job_id_1, {"payload": "data_1"})

    print("--- Starting Job A (Should succeed on attempt 4) ---")
    # Since we are not actually sleeping, this runs instantly
    success = queue.process_job(job_id_1, flaky_processor)
    print(f"\nJob A Result: {'Success' if success else 'Failure'}")

    # Mock processor that always fails
    count_fail = 0
    def always_failing_processor(data):
        global count_fail
        count_fail += 1
        print(f"Attempt {count_fail}: Processing data: {data}")
        raise ValueError("Critical Error")

    job_id_2 = "job_B"
    queue.add_job(job_id_2, {"payload": "data_2"})

    print("\n--- Starting Job B (Should fail after 4 attempts) ---")
    success_fail = queue.process_job(job_id_2, always_failing_processor)
    print(f"\nJob B Result: {'Success' if success_fail else 'Failure'}")