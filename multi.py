from multiprocessing import Process, Queue
import time


def create_data(q, x, t):
    for i in range(1000):
        q.put(x)

if __name__ == '__main__':
    q = Queue()
    p = Process(target=create_data, args=(q, 10, 2, ))
    p.start()
    p = Process(target=create_data, args=(q, 5, 1, ))
    p.start()
    while True:
        print(q.get() ** 2)
        time.sleep(2)
        print(q)


