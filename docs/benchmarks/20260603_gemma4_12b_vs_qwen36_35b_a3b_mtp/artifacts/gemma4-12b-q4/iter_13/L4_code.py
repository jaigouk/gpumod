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

    def process_job(self, job_id, Processor):
        data = None

        # Step 1: Atomically check and remove the job from the queue.
        # By using .pop(), we ensure that only one thread can "claim" 
        # a specific job_id.
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # Step 2: Process the job outside of the lock.
        # This is crucial: we do NOT want to hold the lock while 
        # executing the Processor, otherwise we would block all other 
        # threads from adding or starting other jobs.
        result = Processor(data)

        # Step 3: Atomically store the result.
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# Example Usage:
if __name__ == "__main__":
    def sample_processor(data):
        # Simulate some work
        return f"Processed: {data}"

    queue = JobQueue()
    queue.add_job("job1", "Input Data")

    success = queue.process_job("job1", sample_processor)
    print(f"Success: {success}")
    print(f"Result: {queue.get_result('job1')}")