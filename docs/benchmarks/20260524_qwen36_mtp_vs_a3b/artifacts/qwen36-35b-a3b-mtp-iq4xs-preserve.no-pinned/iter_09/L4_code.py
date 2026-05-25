import threading

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        self.lock = threading.Lock()

    def add_job(self, job_id, data):
        with self.lock:
            self.jobs[job_id] = data
        return job_id

    def process_job(self, job_id, processor):
        with self.lock:
            if job_id not in self.jobs:
                return False
            data = self.jobs.pop(job_id)  # pop is atomic and avoids separate del
            # Note: processor(data) might be slow. If we hold the lock during processing,
            # it will block other threads. The prompt says "BUG: Race condition here - multiple threads can write simultaneously"
            # pointing specifically to the result assignment. But to be safe and correct, 
            # we should only lock around the shared state access, not the processor execution.
            # However, the prompt's comment says: "# BUG: Race condition here - multiple threads can write simultaneously"
            # This implies the race is on self.results and self.jobs.
            # Let's release the lock before calling processor, but we need to be careful.
            # Actually, the standard pattern is to lock only around shared state modifications.
            pass
        result = processor(data)
        with self.lock:
            self.results[job_id] = result
        return True