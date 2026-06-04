import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to synchronize dictionary access
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, Processor):
        # Use the lock to check and remove the job atomically.
        # By using .pop(), we ensure that only one thread can 
        # successfully retrieve the data for a specific job_id.
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # The processing happens OUTSIDE the lock.
        # This allows multiple threads to run different jobs 
        # simultaneously without blocking each other.
        result = Processor(data)

        # Re-acquire the lock to store the result safely.
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
    print(f"Success: {success}, Result: {queue.get_result('task1')}")