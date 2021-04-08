from threading import Thread
import time
import logging
import sys


class DeepSpace(Thread):
    def __init__(self, queue, shuttle):
        # Target changes the function thread.start() will run
        # Otherwise, it will look for a run() method
        Thread.__init__(self, target = self.Function)
        self.queue = queue
        self.shuttle = shuttle
    
    def Function(self):
        for i in range(self.shuttle):
            time.sleep(0.5)
            self.queue.put(i)