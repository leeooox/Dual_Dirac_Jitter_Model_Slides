#-*- coding:utf-8 -*-  

import matplotlib.pyplot as plt
import numpy as np
""" 
NDM问题 
"""  
  

  
class NDM():  
    ''''' n个m面骰子问题 '''  
  
    face=[] #骰面点数  
    count=[] #标志各骰子出现的骰子面标号个数  
  
    times={} #记录NDM各个点数的频次  
  
    def __init__(self, n, m):  
        ''''' n:骰子数 
            m:骰子面数 
            start:骰子面起点值,默认为1 
            end:骰子面终点值,默认为m 
            step:骰子值步进,默认为1 
 
            (end-start+1)/step == m 
            '''  
          
        self.n = n  
        self.m = m  
        self.face = []  
        self.count = []  
        self.times = {}  
        self.xh = []  
  
        for i in range(1, m+1):  
            #face[i]表示第i面的点数  
            self.face.append(i)  
        for i in range(n):  
            #count[i]表示第i个骰子出现的骰子面序号  
            self.count.append(0)  
        self.count[0] = -1  
        #区间[n, m*n]各个点数出现的频数  
        for i in range(n, m*n+1):  
            self.times[i] = 0 #初始化为0  
              
  
    def showProbability(self, decimal):  
        s = self.m**self.n  

        res = [] 
        for i in self.times:  
            #print(i, self.times[i], str(round((self.times[i]*100.0/s),decimal))+"%")  
            res.append([i,float(self.times[i])/s])

        res = np.asarray(res).T
        print res
        plt.plot(res[0],res[1],"-o")
        plt.show()
              
    def funCompute(self):  
        ''''' 计算频次 '''  
        while self.changeCount(0):  
            i = self.getDiceValue()  
            self.times[i] = self.times[i] + 1  
              
        return self.times, self.xh  
  
    def changeCount(self, c):  
        ''''' 每次改变count的值，即一个统计数位器 
            c为进位位 
            '''  
        bAllMax = True #所有骰子都记录到了最大值  
        for i in range(self.n):  
            if self.count[i] != self.m-1:  
                #只要还存在一个骰子还没有记录到最后一面，则表示还没有终止  
                bAllMax = False  
                break  
        if bAllMax:  
            return False #n个骰子全都记录到了最后一面  
  
        bRet = False  
        if c < self.n:  
            if (self.count[c] + 1 > self.m - 1): #有进位  
                self.count[c] = 0  
                bRet = self.changeCount(c+1)  
            else:  
                self.count[c] = self.count[c] + 1  
                return True  
  
        return bRet  
  
    def getDiceValue(self):  
        ''''' 获取投掷n个骰子的点数 '''  
        value = 0  
        #  
        xh = []  
        for i in self.count:  
            xh.append(i)  
        self.xh.append(xh)  
        #  
        for i in self.count:  
            #取得count里记录的每个骰子的面序号  
            #将各个骰子面的点数相加  
            value = value + self.face[i]  
        return value  



c=NDM(7,6)
a,b=c.funCompute()
c.showProbability(3)

