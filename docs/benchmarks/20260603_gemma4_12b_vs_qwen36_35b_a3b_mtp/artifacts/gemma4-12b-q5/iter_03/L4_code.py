import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to synchronize access to shared resources
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, Processor):
        data = None

        # Lock block 1: Atomically "claim" the job
        with self.lock:
            if job_id not in self.jobs:
                return False
            # .pop() removes the item and returns it in one atomic step
            data = self.jobs.pop(job_id)

        # Perform the actual work outside the lock 
        # This allows multiple threads to run processors concurrently
        result = Processor(data)

        # Lock block 2: Atomically save the result
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

    def mock_processor(data):
        return data["value"] * 2

    success = queue.process_job("job1", mock_processor)
    print(f"Success: {success}")
    print(f"Result: {queue.get_result('job1')}")