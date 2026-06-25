import heapq

class JobQueue:
    def __init__(self):
        # Heap stores tuples: (-priority, arrival_index, job_name, job_data)
        # Using negative priority ensures that the highest priority number (2) 
        # results in the smallest stored value (-2), making it the highest priority in the min-heap.
        self.tasks = []
        self.arrival_index = 0

    def add_job(self, name: str, data: dict, priority: int = 0):
        # We store -priority. For priority 2 (Critical), we store -2.
        # This ensures that -2 < -1 < 0, so Critical jobs are processed first.
        heapq.heappush(self.tasks, (-priority, self.arrival_index, name, data))
        self.arrival_index += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self.tasks:
            return None

        # Pop the item with the smallest value (highest priority)
        _, _, name, data = heapq.heappop(self.tasks)

        return name, data