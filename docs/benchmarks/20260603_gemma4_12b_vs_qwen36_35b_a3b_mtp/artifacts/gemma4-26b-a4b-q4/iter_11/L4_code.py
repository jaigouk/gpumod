import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A single lock to protect access to the shared state
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        # 1. Acquire lock to safely check and "claim" the job
        with self.lock:
            if job_id not in self.jobs:
                return False

            # Pop the job immediately so no other thread can process it
            data = self.jobs.pop(job_id)

        # 2. Execute the processor OUTSIDE the lock.
        # This allows other threads to add/process other jobs while this one runs.
        try:
            result = processor(data)
            processing_successful = True
        except Exception as e:
            # In a real system, you might store the error in results
            result = e
            processing_successful = False

        # 3. Acquire lock again to safely write the result
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# --- Example Usage/Test ---
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(0.1)  # Simulate heavy work
        return data * 2

    queue = JobQueue()
    queue.add_job(1, 10)
    queue.add_job(2, 20)

    def worker(jid):
        print(f"Thread {threading.current_thread().name} attempting job {jid}")
        success = queue.process_job(jid, slow_processor)
        print(f"Thread {threading.current_thread().name} job {jid} success: {success}")

    # Create multiple threads attempting to process the same jobs to test race condition
    threads = []
    for i in range(4):
        t1 = threading.Thread(target=worker, args=(1,), name=f"Worker-A-{i}")
        t2 = threading.Thread(target=worker, args=(2,), name=f"Worker-B-{i}")
        threads.extend([t1, t2])

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    print(f"Result for job 1: {queue.get_result(1)}")
    print(f"Result for job 2: {queue.get_result(2)}")