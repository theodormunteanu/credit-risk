# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 09:04:55 2018

@author: theod
"""

#%%
class piecewise_exponential:
    def __init__(self,maturities,intensities):
        r"""
        Inputs:
            
        maturities: the knots around which the default intensities change
        
        intensities: the hazard rate of default
        
        """
        if len(maturities)==len(intensities)-1:
            self.intensities = intensities
            self.maturities = maturities
        else:
            raise TypeError("the number of intensities must be = \
                            with the number of maturities + 1")
    def intensity(self,t):
        indicators = [(self.maturities[i-1]<t and t<=self.maturities[i])*self.intensities[i] \
                      for i in range(1,len(self.maturities))]
        indicators.append((self.maturities[-1]<t)*self.intensities[-1])
        indicators.insert(0,(0<t and t<=self.maturities[0])*self.intensities[0])
        return sum(indicators)
    def survival(self,t):
        r"""
        Output: The probability that the default time is greater than t. 
        
        We compute the integral by quadratures of intensity indicator function. 
        """
        import numpy as np
        import scipy.integrate as integrate
        if t==0:
            return 1
        else:
            f = lambda s:self.intensity(s)
            expo = integrate.quad(f,0,t,limit = 100)[0]
            return np.exp(-expo)
    
    def position(self,t):
        r"""
        We use this function to locate t among the knots (maturities)
        of the piecewise exponential object. 
        """
        if t<=self.maturities[0]:
            return 0
        elif t>self.maturities[-1]:
            return len(self.intensities)-1
        else:
            for i in range(len(self.maturities)):
                if self.maturities[i]<t and t<=self.maturities[i+1]:
                    return i+1
    
    def survival2(self,t):
        r"""
        Same as survival, but we provide directly the analytical formula. 
        It is a faster way of computing the probability of survival. 
        
        """
        import numpy as np
        pos = self.position(t)
        if pos==0:
            return np.exp(-self.intensities[0]*t)
        elif pos==1:
            return np.exp(-self.intensities[0]*self.maturities[0]-self.intensities[1]*(t-self.maturities[0]))
        elif pos == len(self.intensities):
            l1 = self.intensities
            l2 = [self.maturities[i]-self.maturities[i-1] for i in range(1,pos)]
            l2.insert(0,self.maturities[0])
            l2.append(t-self.maturities[-1])
            return np.exp(-np.dot(l1,l2))
        else:
            l1 = [self.intensities[i] for i in range(pos+1)]
            l2 = [self.maturities[i]-self.maturities[i-1] for i in range(1,pos)]
            l2.insert(0,self.maturities[0])
            l2.append(t-self.maturities[pos-1])
            a = np.dot(l1,l2)
            return np.exp(-a)
    
    def cdf(self,t):
        return 1-self.survival(t)
    
    def cdf2(self,t):
        return 1-self.survival2(t)
    def pdf(self,t):
        return self.intensity(t)*self.survival(t)
    def pdf2(self,t):
        return self.intensity(t)*self.survival2(t)
    def survivals(self):
        return [self.survival(self.maturities[i]) for i in range(len(self.maturities))]
    def rvs(self):
        r"""
        Simulate a random variate (Monte Carlo) of the created piecewise exponential 
        object. 
        """
        import numpy as np
        survivs = self.survivals()
        u = np.random.uniform()
        if u<=survivs[-1]:
            return self.maturities[-1]+1/self.intensities[-1] * np.log(survivs[-1]/u)
        elif u>survivs[0]:
            return 1/self.intensities[0] * np.log(1/u)
        else:
            for i in range(len(self.maturities)-1):
                if survivs[i]>=u and u>survivs[i+1]:
                    return self.maturities[i]+1/survivs[i+1]*np.log(survivs[i]/u)
    def rvs2(self,sz = None):
        r"""
        Create sz samples of piecewise exponential model. 
        """
        if sz==None:
            return self.rvs()
        else:
            return [self.rvs() for i in range(sz)]
            
#%%
        
        