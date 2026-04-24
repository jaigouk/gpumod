from collections import deque
import itertools

class JobQueue:
    def __init__(self):
        # Queue stores tuples: (priority, insertion_time, job_name, job_data)
        # We use priority descending (higher number is higher priority) 
        # and insertion_time ascending (FIFO) for sorting.
        self.queue = []
        self.time_counter = itertools.count()

    def add_job(self, job_name: str, job_data: dict, priority: int = 0):
        """
        Adds a job to the queue, maintaining priority and FIFO order.
        Priority: 2 (Critical) > 1 (High) > 0 (Normal).
        """
        insertion_time = next(self.time_counter)
        job_tuple = (priority, insertion_time, job_name, job_data)

        # Find the correct insertion point to maintain sorted order:
        # 1. Primary sort key: Priority (descending)
        # 2. Secondary sort key: Insertion Time (ascending - FIFO)
        
        # We iterate to find the first job that has strictly lower priority 
        # than the new job, or if priorities are equal, we place it after
        # existing jobs of that priority to ensure FIFO.
        
        insert_index = 0
        for i, (p, t, _, _) in enumerate(self.queue):
            # If the existing job has strictly lower priority, insert before it.
            if p < priority:
                insert_index = i
                break
            # If priorities are equal, we continue searching (FIFO requirement).
            # If the existing job has higher priority, we continue searching.
            
            # If we reach the end, the job goes at the end.
            insert_index = i + 1
        
        self.queue.insert(insert_index, job_tuple)


    def get_next_job(self) -> tuple[str, dict] | None:
        """
        Retrieves and removes the highest priority job (which is always at index 0).
        Returns (job_name, job_data).
        """
        if not self.queue:
            return None
        
        # The highest priority job is always at index 0 because of the insertion logic.
        _, _, job_name, job_data = self.queue.pop(0)
        return job_name, job_data

if __name__ == '__main__':
    # Example 1: Standard priority test
    print("--- Example 1: Priority Test ---")
    queue = JobQueue()
    
    # Normal (P=0)
    queue.add_job("normal_A", {"type": "normal"}, priority=0)
    # Critical (P=2)
    queue.add_job("critical_A", {"type": "critical"}, priority=2)
    # High (P=1)
    queue.add_job("high_A", {"type": "high"}, priority=1)
    # Normal (P=0) - Should be after normal_A
    queue.add_job("normal_B", {"type": "normal"}, priority=0)

    # Expected order: critical_A, high_A, normal_A, normal_B

    job = queue.get_next_job()
    print(f"Next job: {job[0]} (Expected: critical_A)")

    job = queue.get_next_job()
    print(f"Next job: {job[0]} (Expected: high_A)")

    job = queue.get_next_job()
    print(f"Next job: {job[0]} (Expected: normal_A)")

    job = queue.get_next_job()
    print(f"Next job: {job[0]} (Expected: normal_B)")

    print(f"Queue empty: {queue.get_next_job() is None}")
    
    # Example 2: FIFO test for equal priorities
    print("\n--- Example 2: FIFO Test ---")
    queue2 = JobQueue()
    
    # Both P=1
    queue2.add_job("high_1", {"type": "high"}, priority=1)
    queue2.add_job("high_2", {"type": "high"}, priority=1)
    
    # P=2
    queue2.add_job("critical_A", {"type": "critical"}, priority=2)
    
    # Another P=1 job inserted later
    queue2.add_job("high_3", {"type": "high"}, priority=1)

    # Expected order: critical_A, high_1, high_2, high_3
    
    job = queue2.get_next_job()
    print(f"Next job: {job[0]} (Expected: critical_A)")

    job = queue2.get_next_job()
    print(f"Next job: {job[0]} (Expected: high_1)")
    
    job = queue2.get_next_job()
    print(f"Next job: {job[0]} (Expected: high_2)")
    
    job = queue2.get_next_job()
    print(f"Next job: {job[0]} (Expected: high_3)")