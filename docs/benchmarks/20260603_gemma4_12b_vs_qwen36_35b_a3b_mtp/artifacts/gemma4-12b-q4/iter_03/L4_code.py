import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to synchronize access to dictionaries
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, Processor):
        # Step 1: Atomically check and remove the job from the queue.
        # This prevents multiple threads from processing the same job_id.
        with self.lock:
            if job_id not in self.jobs:
                return False
            # pop() removes the key and returns the value in one go
            data = self.jobs.pop(job_id)

        # Step 2: Process the data outside of the lock.
        # This allows multiple threads to run 'Processor' concurrently.
        try:
            result = Processor(data)
        except Exception as e:
            # Optional: Handle or log processing errors here
            raise e

        # Step 3: Atomically store the result.
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# Example Usage:
if __name__ == "__main__":
    def mock_processor(data):
        return f"Processed {data}"

    queue = JobQueue()
    queue.add_job("task1", "some_data")

    # This will now safely handle multiple threads calling process_job
    success = queue.process_job("task1", mock_processor)
    print(f"Success: {success}, Result: {queue.get_result('task1')}")