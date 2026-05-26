for attempt in range(1, 4): # 1, 2, 3
             try:
                 return processor()
             except:
                 if attempt < 4:
                     sleep(backoff[attempt-1])
                 else:
                     return False