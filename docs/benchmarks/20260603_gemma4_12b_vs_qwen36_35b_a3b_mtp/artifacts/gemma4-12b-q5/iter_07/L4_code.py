import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to synchronize access to the internal dictionaries
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        data = None

        # Phase 1: Claim the job
        # We use a lock to ensure that only one thread can 
        # "pop" a specific job_id at a time.
        with self.lock:
            if job_id not in self.jobs:
                return False
            # pop() removes the item and returns it atomically
            data = self.jobs.pop(job_id)

        # Phase 2: Process the job
        # We do NOT hold the lock here. This allows multiple threads
        # to run the 'processor' function in parallel for different jobs.
        result = processor(data)

        # Phase 3: Store the result
        with self.lock:
            self.results[job_id] = result

        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# Example usage:
if __name__ == "__main__":
    import time

    def slow_processor(data):
        time.sleep(1) # Simulate heavy work
        return f"Processed: {data}"

    queue = JobQueue()
    queue.add_job("job1", "Task A")
    queue.add_job("job2", "Task B")

    # Simulate multiple threads trying to process the same and different jobs
    def worker(jid):
        success = queue.process_job(jid, slow_processor)
        print(f"Thread finished {jid}: {success}")

    threads = [threading.Thread(target=worker, args=("job1",)) for _ in range(3)]
    threads.append(threading.Thread(target=worker, args=("job2",)))

    for t in threads: t.start()
    for t in threads: t.join()

    print(f"Final Results: {queue.results}")