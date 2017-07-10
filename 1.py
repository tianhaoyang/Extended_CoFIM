#coding:utf8
# import numpy as np
# attr1=[]
# attr2=[]
# attr3=[]
# feature=np.zeros([3,3])
# with open('1.txt','r') as f:
#     for i,line in enumerate(f):
#         a1,a2,a3=line.strip().split('\t')
#         feature[i,0]=a1
#         feature[i,1]=a2
#         feature[i,2]=a3
# # a=np.concatenate((attr1,attr2)).reshape(5,2)
# print a

import multiprocessing as mp
import pathos
from joblib import Parallel, delayed


def func(x):
    print mp.current_process()
    return  x * x



core = mp.cpu_count()
print Parallel(n_jobs=core)(delayed(func)(i) for i in range(10))



# class someClass(object):
#     def __init__(self):
#         self.coe=2
#
#
#     def func(self,x):
#
#         print pathos.helpers.mp.current_process()
#         return self.coe*x*x
#
#     def go(self):
#         core=mp.cpu_count()
#         print Parallel(n_jobs=core)(delayed(self.func)(i) for i in range(10))
        # pool = mp.ProcessingPool(core)

        # print mp._ProcessPool__STATE
        #result = pool.apply_async(self.f, [10])
        #print result.get(timeout=1)
        # print pool.map(self.func, range(10))

# if __name__=='__main__':
    # multiprocessing.freeze_support()
    # a=someClass()
    # a.go()

# import multiprocessing
#
# def f(x):
#     print multiprocessing.current_process()
#     return x * x
#
# if __name__=='__main__':
#     multiprocessing.freeze_support()
#     p = multiprocessing.Pool()
#     print p.map(f, range(6))