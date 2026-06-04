import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        # A lock to synchronize access to the shared dictionaries
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        data = None
        
        # Step 1: Safely check and "claim" the job
        with self.lock:
            if job_id not in self.jobs:
                return False
            # .pop() removes the item and returns it atomically 
            # within our lock context, ensuring no other thread 
            # can grab the same job_id.
            data = self.jobs.pop(job_id)

        # Step 2: Process the job outside the lock.
        # This allows multiple threads to run processors concurrently.
        try:
            result = processor(data)
        except Exception as e:
            print(f"Error processing job {job_id}: {e}")
            return False

        # Step 3: Safely store the result
        with self.lock:
            self.results[job_id] = result
            
        return True

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)

# Example Usage:
if __name__ == "__main__":
    import time

    def mock_processor(data):
        time.sleep(1)  # Simulate heavy work
        return f"Processed: {data}"

    queue = JobQueue()
    queue.add_job("task1", "Data A")
    queue.add_job("task2", "Data B")

    # Simulate two threads processing jobs simultaneously
    def worker(jid):
        success = queue.process_job(jid, mock_processor)
        print(f"Thread finished {jid}: {success}")
        print(f"Result: {queue.get_result(jid)}")

    t1 = threading.Thread(target=worker, args=("task1",))
    t2 = threading.Thread(target=worker, args=("task2",))

    t1.start()
    t2.start()
    t1.join()
    t2.join()