import threading
import queue
import uuid

class JobQueue:
    def __init__(self):
        self._queue = queue.PriorityQueue()
        self._results = {}
        self._lock = threading.Lock()
        self._running = True
        self._seq_counter = 0
        self._seq_lock = threading.Lock()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def add_job(self, fn, *args, priority=5, **kwargs):
        job_id = str(uuid.uuid4())
        with self._seq_lock:
            seq = self._seq_counter
            self._seq_counter += 1
        
        # Negate priority so higher numbers (10) come out first (smaller negative)
        # 10 -> -10, 1 -> -1. -10 < -1.
        # Sequence ensures FIFO for same priority.
        item = (-priority, seq, fn, args, kwargs, job_id)
        self._queue.put(item)
        return job_id

    def get_result(self, job_id):
        with self._lock:
            return self._results.get(job_id)

    def _worker_loop(self):
        while self._running:
            try:
                neg_priority, seq, fn, args, kwargs, job_id = self._queue.get(timeout=0.5)
                try:
                    result = fn(*args, **kwargs)
                except Exception as e:
                    result = e
                with self._lock:
                    self._results[job_id] = result
                self._queue.task_done()
            except queue.Empty:
                continue

    def stop(self):
        self._running = False
        self._worker.join()

if __name__ == "__main__":
    import time

    def sample_task(x):
        time.sleep(1)
        return x * 2

    q = JobQueue()
    job_id = q.add_job(sample_task, 5)
    print(f"Job added: {job_id}")

    while True:
        res = q.get_result(job_id)
        if res is not None:
            print(f"Result: {res}")
            break
        time.sleep(0.1)

    q.stop()
