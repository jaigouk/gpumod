queue = JobQueue()
   job_id = queue.add_job("job1", {"task": "process_data"})
   # After processing...
   result = queue.get_result("job1")