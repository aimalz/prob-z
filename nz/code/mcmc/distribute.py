"""
distribute module sets up for multiprocessing with communication between workers and consumers
"""

import multiprocessing as mp
import sys
import traceback

# class to access data that's been serialized to disk
class distribute_key(object):
    def __init__(self):
        pass

# consumer is the plotter, producer is MCMC calculation
class consumer(object):
    def __init__(self, *args):
        pass
    def loop(self, queue):
        while (True):
            key = queue.get()
            print('got key '+str(key))
            if (key=='done'):
                self.finish()
                return
            self.handle(key)
    def finish(self):
        print "BAD: Generic finish called."

def run_offthread(func, *args):
    proc = mp.Process(target=func, args=args)
    proc.start()
    return proc
def run_offthread_sync(func, *args):
    run_offthread(func, *args).join()

# plot results of one run in loop
def do_consume(ctor, q, args):
    print(ctor,q,args)
    try:
        obj = ctor(**args)
        obj.loop(q)
    except:
        e = sys.exc_info()[0]
        print 'Nested Exception: '
        print traceback.format_exc()
        sys.stdout.flush()
        raise e
def run_consumer(ctor, q, args):
    return run_offthread(do_consume, ctor, q, args)

# class distributes computation over multiple threads
class distribute(object):

    # consumers is list of consumers called and added to queue whenever a new key is complete
    def __init__(self, consumers, start = True, **args):
        self.queues = [mp.Queue() for _ in consumers]
        self.consumer_lambdas = consumers
        self.consumers = [run_consumer(c,q, args) for (c,q) in zip(self.consumer_lambdas, self.queues)]
        self.started = False

    # called by producer when chunk of data has been produced
    def complete_chunk(self, key):
        for q in self.queues:
            q.put(key)
            print('put key '+str(key))

    # called when all producers are done
    def finish(self):
        for q in self.queues:
            q.put('done')
            print('put key '+str('done'))

    def finish_and_wait(self):
        self.finish()
        for c in self.consumers:
            c.join()

    # start the consumers
    def run(self, args):
        if self.started:
            return
        self.started = True
        print('Starting {} threads'.format(len(self.consumers)))
        for t in self.consumers:
            print ('starting: {}'.format(str(t)))
            t.start()
