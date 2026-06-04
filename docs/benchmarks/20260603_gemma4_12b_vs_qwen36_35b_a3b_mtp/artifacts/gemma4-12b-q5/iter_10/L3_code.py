import heapq
    import itertools

    class JobQueue:
        def __init__(self):
            self.queue = []
            self.counter = itertools.count()

        def add_job(self, name, data, priority=0):
            # Since heapq is a min-heap, we negate priority
            # to make higher numbers (2, 1) come out first.
            priority_score = -priority
            count = next(self.counter)
            heapq.heappush(self.queue, (priority_score, count, name, data))

        def get_next_job(self):
            if not self.queue:
                return None
            priority_score, count, name, data = heapq.heappop(self.queue)
            return (name, data)