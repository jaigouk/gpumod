import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to synchronize access to the dictionaries
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # Step 1: Atomically check and "claim" the job
        with self.lock:
            if job_id not in self.jobs:
                return False

            # .pop() removes the item and returns it. 
            # Doing this inside the lock ensures only one thread 
            # can ever process this specific job_id.
            data = self.jobs.pop(job_id)

        # Step 2: Process the job outside the lock.
        # This allows multiple threads to run different jobs simultaneously.
        result = processor(data)

        # Step 3: Atomically store the result
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            # Using the lock ensures we don't read while another 
            # thread is mid-write (important for consistency across different Python impls)
            return self.results.get(job_id)

# Example usage:
if __name__ == "__main__":
    def mock_processor(data):
        return f"Processed {data}"

    queue = JobQueue()
    queue.add_job("job1", "Task Data")

    # This would now be thread-safe
    success = queue.process_job("job1", mock_processor)
    print(f"Success: {success}")
    print(f"Result: {queue.get_result('job1')}")