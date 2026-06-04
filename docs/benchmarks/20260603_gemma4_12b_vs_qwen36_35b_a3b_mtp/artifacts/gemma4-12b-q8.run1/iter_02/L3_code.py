import heapq
    import threading

    class JobQueue:
        def __init__(self):
            self.queue = []
            self.counter = 0
            self.lock = threading.Lock() # Good practice for a queue, though not explicitly asked for thread safety, it's usually implied. But I'll keep it simple unless needed.

        def add_job(self, name: str, data: dict, priority: int = 0):
            # Use -priority because heapq is a min-heap
            # Use self.counter for FIFO order
            heapq.heappush(self.queue, (-priority, self.counter, name, data))
            self.counter += 1

        def get_next_job(self) -> tuple[str, dict] | None:
            if not self.queue:
                return None
            priority, count, name, data = heapq.heappop(self.queue)
            return (name, data)