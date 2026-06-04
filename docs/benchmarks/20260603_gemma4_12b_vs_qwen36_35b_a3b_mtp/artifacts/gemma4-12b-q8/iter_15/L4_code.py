import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to synchronize access to the dictionaries
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        data = None

        # Phase 1: Claim the job
        # We use a lock to ensure only one thread can "pop" the job
        with self.lock:
            if job_id not in self.jobs:
                return False
            # .pop() removes the item and returns it in one atomic step
            data = self.jobs.pop(job_id)

        # Phase 2: Process the job
        # We run the processor OUTSIDE the lock so other threads 
        # can continue adding/processing other jobs concurrently.
        result = processor(data)

        # Phase 3: Store the result
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
    queue.add_job("task1", {"payload": 10})

    def slow_processor(data):
        time.sleep(1) # Simulate work
        return data["payload"] * 2

    # This will now run safely without KeyError or duplicate processing
    success = queue.process_job("task1", slow_processor)
    print(f"Success: {success}, Result: {queue.get_result('task1')}")