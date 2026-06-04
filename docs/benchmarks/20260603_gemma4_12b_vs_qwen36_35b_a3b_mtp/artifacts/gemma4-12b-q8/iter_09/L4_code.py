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
        # 1. Atomically check and remove the job from the queue.
        # We use .pop() inside the lock to ensure only one thread 
        # can "claim" a specific job_id.
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)

        # 2. Run the processor outside of the lock.
        # This prevents a long-running processor from blocking 
        # other threads from adding or retrieving different jobs.
        try:
            result = processor(data)
        except Exception as e:
            # Optional: Handle or log errors from the processor
            print(f"Error processing job {job_id}: {e}")
            return False

        # 3. Atomically write the result.
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# Example usage:
if __name__ == "__main__":
    import time

    queue = JobQueue()
    queue.add_job("task1", {"payload": 10})

    def slow_processor(data):
        time.sleep(1)
        return data["payload"] * 2

    # This will now run safely without race conditions
    success = queue.process_job("task1", slow_processor)
    print(f"Success: {success}, Result: {queue.get_result('task1')}")