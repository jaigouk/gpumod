import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # Initialize a lock to synchronize access to shared dictionaries
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        data = None

        # Phase 1: Atomic "Claim" of the job
        with self.lock:
            if job_id not in self.jobs:
                return False

            # .pop() removes the item and returns it in one atomic step 
            # while holding the lock, ensuring no other thread can grab it.
            data = self.jobs.pop(job_id)

        # Phase 2: Processing
        # We perform the processor work OUTSIDE the lock. 
        # If we held the lock here, only one job could be processed at a time
        # across the entire application, defeating the purpose of threading.
        result = processor(data)

        # Phase 3: Store result
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
        time.sleep(1)  # Simulate heavy work
        return f"Processed: {data}"

    queue = JobQueue()
    queue.add_job("job1", "Task A")
    queue.add_job("job2", "Task B")

    # Simulate two threads trying to process the same job simultaneously
    # Only one should succeed in popping "job1"
    def worker(jid):
        success = queue.process_job(jid, slow_processor)
        print(f"Thread finished {jid}: Success={success}")

    t1 = threading.Thread(target=worker, args=("job1",))
    t2 = threading.Thread(target=worker, args=("job1",)) # Trying same ID
    t3 = threading.Thread(target=worker, args=("job2",))

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    print(f"Final Results: {queue.results}")