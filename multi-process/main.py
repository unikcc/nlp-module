#!/use/bin/env python


"""
@Filanme: generate_pretrain.py
@Author:  unikcc
@Contact: libobo.uk@gmail.com
@Date:    2022/12/17 20:09:19 
"""

from multiprocess import Process
import time
import random


def startx(thread_id):
    s = random.random() * 5
    time.sleep(s)
    print("Thread-{} start".format(thread_id))
    s = random.random() * 1
    time.sleep(s)
    print("Thread-{} end".format(thread_id))

def run():
    thread_num = 10

    ps = []
    for i in range(thread_num):
        p = Process(target=startx, args=(i,))
        ps.append(p)
        p.start()
    
    for p in ps:
        p.join()


run()