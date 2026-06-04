import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to synchronize access to the shared dictionaries
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, Processor):
        # Step 1: Atomically "claim" the job.
        # By popping it now, we guarantee that no other thread 
        # can pull the same job_id simultaneously.
        with self._lock:
            data = self.jobs.pop(job_id, None)
            if data is None:
                return False

        # Step 2: Run the processor outside of the lock.
        # This allows multiple threads to run processors in parallel.
        try:
            result = Processor(data)
        except Exception as e:
            # Optional: Handle or log exception if necessary
            raise e

        # Step 3: Atomically write the result.
        with self._lock:
            self.results[job_id] = result
            return True

    def get_result(self, job_id):
        with self._lock:
            return self.results.get(job_id)

# Example usage:
if __name__ == "__main__":
    import time

    queue = JobQueue()
    queue.add_job("task1", {"value": 10})

    def simple_processor(data):
        time.sleep(1) # Simulate heavy work
        return data["value"] * 2

    # This will now be thread-safe
    success = queue.process_job("task1", simple_processor)
    print(f"Success: {success}, Result: {queue.get_result('task1')}")