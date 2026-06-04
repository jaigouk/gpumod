import heapq

    class JobQueue:
        def __init__(self):
            self.heap = []
            self.count = 0

        def add_job(self, name: str, data: dict, priority: int = 0):
            # heapq is a min-heap.
            # We want highest priority first, so we use -priority.
            # We want FIFO for same priority, so we use count.
            heapq.heappush(self.heap, (-priority, self.count, name, data))
            self.count += 1

        def get_next_job(self) -> tuple[str, dict] | None:
            if not self.heap:
                return None
            _, _, name, data = heapq.heappop(self.heap)
            return (name, data)