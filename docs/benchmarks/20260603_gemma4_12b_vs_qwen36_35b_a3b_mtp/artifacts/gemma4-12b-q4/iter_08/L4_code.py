import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to synchronize access to the shared dictionaries
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Step 1: Atomically "claim" the job.
        # By using .pop(), we ensure that only one thread successfully 
        # retrieves the data. Any other thread trying to process the 
        # same ID will fail the membership check.
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # Step 2: Execute the processor.
        # We do NOT hold the lock here. If processor(data) takes 10 seconds,
        # we don't want to block other threads from adding or processing other jobs.
        result = processor(data)

        # Step 3: Store the result.
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# Example Usage:
if __name__ == "__main__":
    def sample_processor(data):
        return f"Processed: {data}"

    queue = JobQueue()
    queue.add_job("task1", "Some Data")

    # This will now safely handle multiple threads trying to access "task1"
    success = queue.process_job("task1", sample_processor)
    print(f"Success: {success}, Result: {queue.get_result('task1')}")