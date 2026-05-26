import heapq
    import time

    class JobQueue:
        def __init__(self):
            self.heap = []
            self.counter = 0

        def add_job(self, name: str, data: dict, priority: int = 0):
            # Store as (-priority, counter, name, data)
            # Min-heap: smaller value pops first.
            # -2 (priority 2) < -1 (priority 1) < -0 (priority 0)
            heapq.heappush(self.heap, (-priority, self.counter, name, data))
            self.counter += 1

        def get_next_job(self) -> tuple[str, dict] | None:
            if not self.heap:
                return None
            # Pop smallest (highest priority)
            priority_neg, seq, name, data = heapq.heappop(self.heap)
            return (name, data)