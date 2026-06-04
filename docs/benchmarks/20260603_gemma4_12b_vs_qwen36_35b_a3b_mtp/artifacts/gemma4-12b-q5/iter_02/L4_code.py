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
        data = None

        # Critical Section 1: Check existence and remove the job
        # We pop the job immediately so no other thread can process it
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # The processor is run outside the lock.
        # This allows multiple jobs to be processed in parallel 
        # by different threads without blocking each other.
        result = processor(data)

        # Critical Section 2: Write the result
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# Example Usage:
if __name__ == "__main__":
    queue = JobQueue()
    queue.add_job("task1", {"value": 10})

    def simple_processor(data):
        return data["value"] * 2

    success = queue.process_job("task1", simple_processor)
    print(f"Success: {success}")
    print(f"Result: {queue.get_result('task1')}")