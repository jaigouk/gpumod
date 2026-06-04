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
        data = None

        # Step 1: Atomically check for existence and "claim" the job
        with self.lock:
            if job_id not in self.jobs:
                return False

            # .pop() removes the item and returns it in one atomic step
            # This ensures no other thread can pick up this specific job_id
            data = self.jobs.pop(job_id)

        # Step 2: Process the data outside the lock.
        # This allows other threads to continue adding/processing other jobs
        # while this specific thread is busy with a potentially slow processor.
        result = processor(data)

        # Step 3: Store the result safely
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# Example Usage:
if __name__ == "__main__":
    import time

    queue = JobQueue()
    queue.add_job("job1", "some data")

    def slow_processor(data):
        time.sleep(1) # Simulate heavy work
        return f"Processed: {data}"

    # This would have caused a race condition/error in the original code
    # if called by multiple threads simultaneously.
    success = queue.process_job("job1", slow_processor)
    print(f"Success: {success}")
    print(f"Result: {queue.get_result('job1')}")