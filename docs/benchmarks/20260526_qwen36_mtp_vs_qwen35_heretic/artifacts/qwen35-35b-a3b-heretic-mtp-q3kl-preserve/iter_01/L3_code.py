import heapq
    from itertools import count

    class JobQueue:
        def __init__(self):
            self.heap = []
            self.counter = count()

        def add_job(self, name: str, data: dict, priority: int = 0) -> None:
            seq = next(self.counter)
            # Negate priority so higher priority (2) becomes smaller (-2) in min-heap
            heapq.heappush(self.heap, (-priority, seq, name, data))

        def get_next_job(self) -> tuple[str, dict] | None:
            if not self.heap:
                return None
            _, _, name, data = heapq.heappop(self.heap)
            return (name, data)