import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Create a lock to protect shared resources (self.jobs and self.results)
        self._lock = threading.Lock()

    def add_job(self, job_id, data):
        with self._lock:
            self.jobs[job_id] = data
            return job_id

    def process_job(self, job_id, processor):
        """
        Processes a single job. The entire operation (read, process, write result, delete job)
        must be protected by the lock to prevent race conditions.
        """
        with self._lock:
            # 1. Check existence and retrieve data (Read)
            if job_id not in self.jobs:
                return False

            data = self.jobs[job_id]

            # Note: We could potentially release the lock here if the processor is very long,
            # but in this simple structure, holding the lock ensures the state update is consistent.
            result = processor(data)

            # 2. Write result
            self.results[job_id] = result

            # 3. Delete job from pending queue
            del self.jobs[job_id]

            return True

    def get_result(self, job_id):
        # Lock is still useful when reading shared state to ensure we don't read while
        # another thread is deleting or modifying the dictionary structure.
        with self._lock:
            return self.results.get(job_id)

# --- Example Usage ---
if __name__ == '__main__':
    queue = JobQueue()

    def my_processor(data):
        # Simulate some processing time
        import time
        time.sleep(0.01) 
        return f"Processed: {data}"

    # Add jobs
    job1_id = queue.add_job("J1", "Data A")
    job2_id = queue.add_job("J2", "Data B")

    # Simulate concurrent processing
    threads = []

    def worker(job_id):
        success = queue.process_job(job_id, my_processor)
        print(f"Job {job_id} processed: {'Success' if success else 'Failed'}")

    t1 = threading.Thread(target=worker, args=(job1_id,))
    t2 = threading.Thread(target=worker, args=(job2_id,))

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    # Verify results
    print(f"\nResult for J1: {queue.get_result(job1_id)}")
    print(f"Result for J2: {queue.get_result(job2_id)}")
    print(f"Jobs remaining in queue: {list(queue.jobs.keys())}")