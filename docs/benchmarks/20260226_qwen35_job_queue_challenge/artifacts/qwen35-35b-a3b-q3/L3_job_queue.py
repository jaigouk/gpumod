import threading
import queue
import uuid
import time
from typing import Any, Callable, Optional

class JobQueue:
    def __init__(self, num_workers: int = 4):
        self._queue = queue.PriorityQueue()
        self._results = {}
        self._lock = threading.Lock()
        self._workers = []
        self._stop_event = threading.Event()
        self._counter = 0
        self._counter_lock = threading.Lock()

        for _ in range(num_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self._workers.append(t)

    def add_job(self, fn: Callable, *args, priority: int = 5, **kwargs) -> str:
        job_id = str(uuid.uuid4())
        
        with self._counter_lock:
            seq = self._counter
            self._counter += 1
            
        # Invert priority so higher priority (10) is processed before lower (1)
        # PriorityQueue pops smallest item first.
        # We use (-priority, sequence) to ensure FIFO for same priority.
        item = (-priority, seq, fn, args, kwargs, job_id)
        self._queue.put(item)
        return job_id

    def get_result(self, job_id: str) -> Optional[Any]:
        with self._lock:
            return self._results.get(job_id)

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.5)
                # item is (-priority, seq, fn, args, kwargs, job_id)
                _, _, fn, args, kwargs, job_id = item
                try:
                    result = fn(*args, **kwargs)
                except Exception as e:
                    result = e
                with self._lock:
                    self._results[job_id] = result
                self._queue.task_done()
            except queue.Empty:
                continue

    def shutdown(self):
        self._stop_event.set()
        for t in self._workers:
            t.join()

if __name__ == "__main__":
    def example_task(x):
        import time
        time.sleep(0.5)
        return x * 2

    jq = JobQueue()
    job_id = jq.add_job(example_task, 10, priority=10)
    
    while jq.get_result(job_id) is None:
        import time
        time.sleep(0.1)
        
    print(f"Result: {jq.get_result(job_id)}")
    jq.shutdown()
