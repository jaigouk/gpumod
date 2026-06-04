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

        # CRITICAL SECTION: Check and "claim" the job
        with self.lock:
            if job_id not in self.jobs:
                return False

            # By using .pop(), we remove the job from the 'jobs' dict 
            # immediately. This ensures that only one thread can successfully 
            # "claim" this job_id to process it.
            data = self.jobs.pop(job_id)

        # PROCESSING SECTION:
        # We perform the actual processing OUTSIDE the lock.
        # This allows other threads to continue adding or processing 
        # other jobs while this thread is busy working.
        result = Processor(data)

        # FINALIZATION SECTION:
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# Example usage:
if __name__ == "__main__":
    def simple_processor(data):
        # Simulate work
        return f"Processed {data}"

    queue = JobQueue()
    queue.add_job("job1", "Task Data")

    success = queue.process_job("job1", simple_processor)
    print(f"Success: {success}")
    print(f"Result: {queue.get_result('job1')}")