from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        # Stores job data: {job_id: {'data': dict, 'attempts': int, 'last_delay': float}}
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.MAX_ATTEMPTS = 4

    def add_job(self, job_id: str, data: dict):
        """Adds a job to the queue."""
        self.jobs[job_id] = {
            'data': data,
            'attempts': 0,
            'last_delay': 0.0
        }

    def process_job(self, job_id: str, processor: Callable) -> bool:
        """
        Processes a job with a retry mechanism and exponential backoff.

        :param job_id: The unique identifier of the job.
        :param processor: The callable function (data: dict) to execute.
        :return: True if successful, False if all attempts failed.
        """
        if job_id not in self.jobs:
            raise KeyError(f"Job {job_id} not found in the queue.")

        job_state = self.jobs[job_id]
        data = job_state['data']
        attempts = job_state['attempts']

        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            job_state['attempts'] = attempt

            try:
                # 1. Call processor
                processor(data)
                # Success
                return True

            except Exception as e:
                print(f"Job {job_id} attempt {attempt} failed: {e}")

                # 4. Do NOT actually sleep — record the delays as data.
                # 3. Use exponential backoff (1s, 2s, 4s)
                if attempt < self.MAX_ATTEMPTS:
                    delay = 2 ** (attempt - 1)  # 1, 2, 4
                    job_state['last_delay'] = delay
                    print(f"Retrying job {job_id} in {delay} seconds...")
                    # Simulation of waiting (omitted as per requirements)

        # 5. Return False if all 4 attempts raise
        return False

if __name__ == '__main__':
    # --- Example Usage ---

    # 1. Define a processor that fails initially but succeeds on the 3rd try
    attempt_counter = 0

    def unstable_processor(data: dict) -> None:
        global attempt_counter
        attempt_counter += 1
        print(f"Executing processor with data: {data} (Attempt {attempt_counter})")

        # Fail first two attempts
        if attempt_counter < 3:
            raise ValueError(f"Simulated error on attempt {attempt_counter}")

        # Succeed on third attempt
        print("Processor succeeded!")

    # 2. Initialize the queue
    queue = JobQueue()
    queue.add_job("task_A", {"input": "user_data_1"})

    print("--- Starting Job Processing Test (Task A) ---")
    success = queue.process_job("task_A", unstable_processor)

    print("\n--- Results ---")
    print(f"Task A successful? {success}")

    # Reset counter for a failure test
    attempt_counter = 0

    # 3. Define a processor that always fails
    def always_failing_processor(data: dict) -> None:
        raise RuntimeError("Permanent system failure")

    queue.add_job("task_B", {"input": "user_data_2"})

    print("\n--- Starting Job Processing Test (Task B) ---")
    success_fail = queue.process_job("task_B", always_failing_processor)

    print("\n--- Results ---")
    print(f"Task B successful? {success_fail}")

    # Verification of internal state tracking (Task B should have failed after 4 attempts)
    # print("\nInternal Job State for Task B:", queue.jobs.get("task_B"))