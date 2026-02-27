import threading
import queue
import uuid
from typing import Any, Callable, Dict, Optional

class JobQueue:
    def __init__(self, num_workers: int = 1):
        self.job_queue = queue.PriorityQueue()
        self.results: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.workers = []
        self.stop_event = threading.Event()
        self.sequence_counter = 0

        for _ in range(num_workers):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self.workers.append(t)

    def add_job(self, fn: Callable, *args, priority: int = 5, **kwargs) -> str:
        job_id = str(uuid.uuid4())
        with self.lock:
            self.results[job_id] = {'status': 'pending', 'result': None}
            self.sequence_counter += 1
            seq = self.sequence_counter
        
        # Clamp priority to 1-10 range to ensure consistent behavior
        priority = max(1, min(10, priority))
        
        # Use negative priority so higher priority (10) becomes smaller value (-10)
        # and is popped first by PriorityQueue.
        # Use sequence counter for FIFO ordering within same priority.
        self.job_queue.put((-priority, seq, job_id, fn, args, kwargs))
        return job_id

    def get_result(self, job_id: str) -> Optional[Any]:
        with self.lock:
            if job_id not in self.results:
                return None
            job_info = self.results[job_id]
            if job_info['status'] == 'completed':
                return job_info['result']
            elif job_info['status'] == 'failed':
                return job_info['result']
            else:
                return None

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                # Unpack: (-priority, seq, job_id, fn, args, kwargs)
                _, _, job_id, fn, args, kwargs = self.job_queue.get(timeout=0.1)
                try:
                    result = fn(*args, **kwargs)
                    with self.lock:
                        if job_id in self.results:
                            self.results[job_id] = {'status': 'completed', 'result': result}
                except Exception as e:
                    with self.lock:
                        if job_id in self.results:
                            self.results[job_id] = {'status': 'failed', 'result': e}
                finally:
                    self.job_queue.task_done()
            except queue.Empty:
                continue

    def shutdown(self):
        self.stop_event.set()
        for t in self.workers:
            t.join()

if __name__ == "__main__":
    import time

    def slow_task(x):
        time.sleep(1)
        return x * 2

    jq = JobQueue()
    job_id = jq.add_job(slow_task, 5)
    print(f"Added job: {job_id}")
    
    time.sleep(0.1)
    print(f"Result (pending): {jq.get_result(job_id)}")
    
    time.sleep(2)
    print(f"Result (completed): {jq.get_result(job_id)}")
    
    jq.shutdown()
