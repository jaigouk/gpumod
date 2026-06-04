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

    def process_job(self, job_id, processor):
        data = None
        
        # Phase 1: Atomically check and "claim" the job
        with self._lock:
            if job_id not in self.jobs:
                return False
            
            # Pop the job immediately so no other thread can 
            # pick it up while we are processing it.
            data = self.jobs.pop(job_id)

        # Phase 2: Process the job outside the lock.
        # This allows multiple jobs to be processed in parallel.
        result = processor(data)

        # Phase 3: Atomically store the result.
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
    queue.add_job("task1", 10)
    queue.add_job("task2", 20)

    def slow_processor(n):
        time.sleep(1)  # Simulate work
        return n * 2

    # Simulate concurrent processing
    t1 = threading.Thread(target=queue.process_job, args=("task1", slow_processor))
    t2 = threading.Thread(target=queue.process_job, args=("task2", slow_processor))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print(f"Result 1: {queue.get_result('task1')}")
    print(f"Result 2: {queue.get_result('task2')}")