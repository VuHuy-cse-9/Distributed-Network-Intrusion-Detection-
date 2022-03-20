import logging
import threading
import time

def thread_function(name):
    logging.info("Thread %s: starting", name)
    time.sleep(2)
    logging.info("Thread %s: finishing", name)
    
if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
    threads = []
    for index in range(3):
        logging.info(f"Main: before creating thread {index}")
        x = threading.Thread(target=thread_function, args=(index,), daemon=True)
        threads.append(x)
        x.start()
    
    for index, thread in enumerate(threads):
        logging.info("Main : wait for the thread %d to finish", index)
        thread.join()
        logging.info("Main : thread %d done", index)
    