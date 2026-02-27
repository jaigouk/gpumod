import threading
import queue
import uuid
import time

class JobQueue:
    def __init__(self, num_workers=4):
        self.job_queue = queue.PriorityQueue()
        self.results = {}
        self.results_lock = threading.Lock()
        self.workers = []
        self.stop_event = threading.Event()
        self._counter = 0
        self._counter_lock = threading.Lock()

        for _ in range(num_workers):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self.workers.append(t)

    def add_job(self, fn, *args, priority=5, **kwargs):
        job_id = str(uuid.uuid4())
        
        with self._counter_lock:
            counter = self._counter
            self._counter += 1
        
        # Invert priority so higher number = lower value in PQ (since PQ pops smallest first)
        # Priority 10 -> -10, Priority 1 -> -1. -10 < -1, so 10 comes first.
        pq_item = (-priority, counter, job_id, fn, args, kwargs)
        
        self.job_queue.put(pq_item)
        
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
                # PQ returns tuple: (-priority, counter, job_id, fn, args, kwargs)
                item = self.job_queue.get(timeout=1)
                # Unpack
                # priority = -item[0] # Not needed for execution
                # counter = item[1]
                job_id = item[2]
                fn = item[3]
                args = item[4]
                kwargs = item[5]
                
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
