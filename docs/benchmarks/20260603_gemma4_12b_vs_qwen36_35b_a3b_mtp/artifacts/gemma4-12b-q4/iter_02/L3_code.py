import heapq
        import itertools

        class JobQueue:
            def __init__(self):
                self.queue = []
                self.counter = itertools.count()

            def add_job(self, name: str, data: dict, priority: int = 0):
                # heapq is a min-heap. To get max priority first, negate priority.
                heapq.heappush(self.queue, (-priority, next(self.counter), name, data))

            def get_next_job(self) -> tuple[str, dict] | None:
                if not self.queue:
                    return None
                _, _, name, data = heapq.heappop(self.queue)
                return name, data