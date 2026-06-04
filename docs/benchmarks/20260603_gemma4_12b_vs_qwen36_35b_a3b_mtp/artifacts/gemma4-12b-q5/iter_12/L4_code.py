import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to synchronize access to shared dictionaries
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, Processor):
        data = None

        # Step 1: Atomically check and "claim" the job
        with self.lock:
            if job_id not in self.jobs:
                return False

            # pop() removes the item and returns it in one atomic step
            # This ensures no other thread can process this specific job_id
            data = self.jobs.pop(job_id)

        # Step 2: Run the processor outside the lock.
        # This allows other threads to process other jobs simultaneously.
        try:
            result = Processor(data)
        except Exception as e:
            # Optional: Handle errors here if necessary
            raise e

        # Step 3: Store the result safely
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# Example Usage:
if __name__ == "__main__":
    queue = JobQueue()
    queue.add_job("job1", {"value": 10})

    def example_processor(data):
        return data["value"] * 2

    success = queue.process_job("job1", example_processor)
    print(f"Success: {success}, Result: {queue.get_result('job1')}")