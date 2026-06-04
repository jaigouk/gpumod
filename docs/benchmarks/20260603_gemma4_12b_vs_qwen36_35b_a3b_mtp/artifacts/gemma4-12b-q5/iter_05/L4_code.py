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

    def process_job(self, job_id, processor):
        data = None

        # Step 1: Atomically check and "claim" the job
        with self.lock:
            if job_id not in self.jobs:
                return False
            # pop() retrieves the value and removes the key in one step
            data = self.jobs.pop(job_id)

        # Step 2: Execute the processor outside of the lock.
        # This allows multiple threads to run different processors 
        # simultaneously without blocking each other.
        result = processor(data)

        # Step 3: Atomically store the result
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# Example usage:
if __name__ == "__main__":
    queue = JobQueue()
    queue.add_job("1", "Some Data")

    def sample_processor(data):
        return f"Processed: {data}"

    success = queue.process_job("1", sample_processor)
    print(f"Success: {success}")
    print(f"Result: {queue.get_result('1')}")