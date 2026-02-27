import threading
import queue
import uuid
import time

class JobQueue:
    def __init__(self, num_workers=4):
        self.job_queue = queue.Queue()
        self.results = {}
        self.results_lock = threading.Lock()
        self.workers = []
        self.stop_event = threading.Event()

        for _ in range(num_workers):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self.workers.append(t)

    def add_job(self, fn, *args, **kwargs):
        job_id = str(uuid.uuid4())
        self.job_queue.put((job_id, fn, args, kwargs))
        with self.results_lock:
            self.results[job_id] = {'status': 'pending', 'result': None}
        return job_id

    def get_result(self, job_id):
        with self.results_lock:
            if job_id not in self.results:
                return None
            job_info = self.results[job_id]
            if job_info['status'] == 'done':
                return job_info['result']
            return None

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                job_id, fn, args, kwargs = self.job_queue.get(timeout=1)
                try:
                    result = fn(*args, **kwargs)
                    with self.results_lock:
                        self.results[job_id] = {'status': 'done', 'result': result}
                except Exception as e:
                    with self.results_lock:
                        self.results[job_id] = {'status': 'done', 'result': e}
                finally:
                    self.job_queue.task_done()
            except queue.Empty:
                continue

    def shutdown(self):
        self.stop_event.set()
        for t in self.workers:
            t.join()

if __name__ == "__main__":
    def slow_task(x):
        time.sleep(0.5)
        return x * 2

    jq = JobQueue()
    job_id = jq.add_job(slow_task, 5)
    
    while jq.get_result(job_id) is None:
        time.sleep(0.1)
        
    result = jq.get_result(job_id)
    print(f"Result: {result}")
    
    jq.shutdown()
