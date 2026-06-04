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

        # Step 1: Atomic Check and Retrieval
        # We lock only long enough to "claim" the job.
        with self.lock:
            # .pop() removes the item and returns it, or None if not found
            data = self.jobs.pop(job_id, None)

        if data is None:
            return False

        # Step 2: Processing
        # We perform the actual work OUTSIDE the lock so other threads
        # aren't blocked while this specific job is being processed.
        result = Processor(data)

        # Step 3: Atomic Result Writing
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# Example Usage:
if __name__ == "__main__":
    def slow_processor(data):
        import time
        time.sleep(1) # Simulate work
        return f"Processed: {data}"

    queue = JobQueue()
    queue.add_job("job1", "some data")

    # This will now safely return True for the first thread that grabs it
    # and False for any subsequent threads trying to grab the same ID.
    success = queue.process_job("job1", slow_processor)
    print(f"Success: {success}")
    print(f"Result: {queue.get_result('job1')}")