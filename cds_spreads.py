# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 14:19:27 2020

@author: Theodor
"""


class cds_spreads:
    r"""
    ------------------------
    Estimated spread of a deferred coupon CDS. 
    ------------------------
    
    
    Standard parameters: 
        
    T: float. The CDS lifetime
    
    R: float. Recovery rate
    
    defer: float. The time while coupons are exempted
    
    freq: int. frequency of payments. By default is 0, meaning a continuous payment.
    
    We check the estimation according to the following models:
    
    ----------------------------------------------
    
    == Exponential: Expo
    
    == Piecewise exponential 
       
    == Incomplete logistic distribution.
    $$S(x) = 2/(1+e^{-x}),x>0$$ 
    
    == Log-Logistic distribution
    $$S(x) = 1/(1+(x/{\alpha})^{\beta},x>0$$
    
    == Log-normal distribution
    
    == Weibull distribution: 
    """
    def __init__(self,T,R,freq = 0):
        self.T=T
        self.R = R
        self.freq = freq
    
    def expo(self,lbd,r=0,t0=0,RPV01 = False,prob_dist = False):
        import numpy as np
        f = lambda t:np.exp(-lbd*t)*lbd
        S = lambda t:np.exp(-lbd*t)
        D = lambda t:np.exp(-r*t)
        T,R,freq=self.T,self.R,self.freq
        import scipy.integrate as integ
        q1 = integ.quad(lambda s:f(s)*D(s-t0)/S(t0)*(s>t0),t0,T)[0]
        if freq==0:
            q2 = integ.quad(lambda s:S(s)*D(s-t0)/S(t0)*(s>t0),t0,T)[0]
        else:
            dt = 1/freq
            times = [x for x in np.linspace(1/freq,T,int(T*freq)) if x>t0]
            q2 = dt * sum([S(x)/S(t0)*np.exp(-r*(x-t0)) for x in times])
        return (1-R)*q1/q2 if RPV01 == False else ((1-R)*q1/q2,q2)
        
    
    def piece_expo(self,maturities,intensities,r=0,t0=0,RPV01 = False,prob_dist = False):
        import numpy as np
        from piecewise_expo import piecewise_exponential
        T,R,freq=self.T,self.R,self.freq
        pe = piecewise_exponential(maturities,intensities)
        f = lambda t:pe.pdf2(t)
        S = lambda t:pe.survival2(t)
        D = lambda t:np.exp(-r*t)
        import scipy.integrate as integ
        q1 = integ.quad(lambda s:f(s)*D(s-t0)/S(t0)*(s>t0),t0,T)[0]
        if freq==0:
            q2 = integ.quad(lambda s:S(s)*D(s-t0)/S(t0)*(s>t0),t0,T)[0]
        else:
            dt = 1/freq
            times = [x for x in np.linspace(1/freq,T,int(T*freq)) if x>t0]
            q2 = dt * sum([S(x)/S(t0)*np.exp(-r*(x-t0)) for x in times])
        
        return (1-R)*q1/q2 if RPV01 == False else ((1-R)*q1/q2,q2)
        
    
    def Weibull(self,lbd,gamma,r=0,t0=0,RPV01 = False,prob_dist = False):
        import numpy as np
        T,R,freq=self.T,self.R,self.freq
        f= lambda t:lbd*gamma*t**(gamma-1)*np.exp(-lbd*t**gamma)
        S = lambda t:np.exp(-lbd*t**gamma)
        D = lambda t:np.exp(-r*t)
        import scipy.integrate as integ
        q1 = integ.quad(lambda s:f(s)*D(s-t0)/S(t0)*(s>t0),t0,T)[0]
        if freq==0:
            q2 = integ.quad(lambda s:S(s)*D(s-t0)/S(t0)*(s>t0),t0,T)[0]
        else:
            dt = 1/freq
            times = [x for x in np.linspace(1/freq,T,int(T*freq)) if x>t0]
            q2 = dt * sum([S(x)/S(t0)*np.exp(-r*(x-t0)) for x in times])
        
        return (1-R)*q1/q2 if RPV01 == False else ((1-R)*q1/q2,q2)
        
    def piecewise_Weibull(self,maturities,intensities,gamma,r=0,t0=0,RPV01 = False):
        import numpy as np
        from piecewise_weibull import piecewise_Weibull
        pw = piecewise_Weibull(maturities,intensities,gamma)
        f= lambda t:pw.pdf(t)
        S = lambda t:pw.survival(t)
        D = lambda t:np.exp(-r*t)
        T,R,freq=self.T,self.R,self.freq
        import scipy.integrate as integ
        q1 = integ.quad(lambda s:f(s)*D(s-t0)/S(t0)*(s>t0),t0,T)[0]
        if freq==0:
            q2 = integ.quad(lambda s:S(s)*D(s-t0)/S(t0)*(s>t0),t0,T)[0]
        else:
            dt = 1/freq
            times = [x for x in np.linspace(1/freq,T,int(T*freq)) if x>t0]
            q2 = dt * sum([S(x)/S(t0)*np.exp(-r*(x-t0)) for x in times])
        return (1-R)*q1/q2 if RPV01 == False else ((1-R)*q1/q2,q2)
    
    def Gompertz(self,eta,b,r=0,t0=0,RPV01 = False):
        import numpy as np
        f= lambda t:b*eta*np.exp(eta+b*t-eta*np.exp(b*t))
        S = lambda t:np.exp(-eta*(np.exp(b*t)-1))
        D = lambda t:np.exp(-r*t)
        T,R,freq=self.T,self.R,self.freq
        import scipy.integrate as integ
        q1 = integ.quad(lambda s:f(s)*D(s-t0)/S(t0)*(s>t0),t0,T)[0]
        if freq==0:
            q2 = integ.quad(lambda s:S(s)*D(s-t0)/S(t0)*(s>t0),t0,T)[0]
        else:
            dt = 1/freq
            times = [x for x in np.linspace(1/freq,T,int(T*freq)) if x>t0]
            q2 = dt * sum([S(x)/S(t0)*np.exp(-r*(x-t0)) for x in times])
        return (1-R)*q1/q2 if RPV01 == False else ((1-R)*q1/q2,q2)
    
    def incomplete_logistic(self,lbd,r=0,t0=0,RPV01 = False,prob_dist = False):
        import numpy as np
        T,R,freq=self.T,self.R,self.freq
        f = lambda x:2*lbd*np.exp(-lbd*x)/(1+np.exp(-lbd*x))**2 *(x>=0)
        S = lambda x:2/(1+np.exp(-lbd*x))
        D = lambda t:np.exp(-r*t)
        import scipy.integrate as integ
        q1 = integ.quad(lambda s:f(s)*D(s-t0)/S(t0)*(s>t0),t0,T)[0]
        if freq==0:
            q2 = integ.quad(lambda s:S(s)*D(s-t0)/S(t0)*(s>t0),t0,T)[0]
        else:
            dt = 1/freq
            times = [x for x in np.linspace(1/freq,T,int(T*freq)) if x>t0]
            q2 = dt * sum([S(x)/S(t0)*np.exp(-r*(x-t0)) for x in times])
        
        return (1-R)*q1/q2 if RPV01 == False else ((1-R)*q1/q2,q2)
        
    
    def log_logistic(self,alpha,beta,r=0,t0=0,RPV01 = False,prob_dist = False):
        r"""
        Inputs:
        alpha: float. Scale parameter of the distribution.
        
        beta: float. Shape parameter of the distribution.
        """
        import numpy as np
        T,R,freq=self.T,self.R,self.freq
        f = lambda t:beta/alpha * (t/alpha)**(beta-1)/(1+(t/alpha)**beta)**2
        S = lambda t:1/(1+(t/alpha)**beta)
        D = lambda t:np.exp(-r*t)
        import scipy.integrate as integ
        q1 = integ.quad(lambda s:f(s)*D(s-t0)/S(t0)*(s>t0),t0,T)[0]
        if freq==0:
            q2 = integ.quad(lambda s:S(s)*D(s-t0)/S(t0)*(s>t0),t0,T)[0]
        else:
            dt = 1/freq
            times = [x for x in np.linspace(1/freq,T,int(T*freq)) if x>t0]
            q2 = dt * sum([S(x)/S(t0)*np.exp(-r*(x-t0)) for x in times])
        return (1-R)*q1/q2 if RPV01 == False else ((1-R)*q1/q2,q2)
        
    
    def log_normal(self,lbd,gamma,r=0,t0=0,RPV01 = False,prob_dist = False):
        import numpy as np
        import scipy.stats as stats
        T,R,freq=self.T,self.R,self.freq
        f = lambda t:stats.norm.pdf(gamma*np.log(lbd*t))*(gamma*lbd)/t
        S = lambda t:stats.norm.sf(gamma*np.log(lbd*t))
        D = lambda t:np.exp(-r*t)
        import scipy.integrate as integ
        q1 = integ.quad(lambda s:f(s)*D(s-t0)/S(t0)*(s>t0),t0,T)[0]
        if freq==0:
            q2 = integ.quad(lambda s:S(s)*D(s-t0)/S(t0)*(s>t0),t0,T)[0]
        else:
            dt = 1/freq
            times = [x for x in np.linspace(1/freq,T,int(T*freq)) if x>t0]
            q2 = dt * sum([S(x)/S(t0)*np.exp(-r*(x-t0)) for x in times])
        return (1-R)*q1/q2 if RPV01 == False else ((1-R)*q1/q2,q2)
        

#%%  
class cds_values:
    r"""
    Value of the CDS to the protection buyer. 
    """
    def __init__(self,T,R,c,N,freq = 0):
       self.T=T
       self.R = R
       self.freq = freq
       self.c = c
       self.N = N
    
    def expo(self,lbd,r=0,t0=0):
       T,R,freq,N = self.T,self.R,self.freq,self.N
       cds = cds_spreads(T,R,freq)
       spread,RPV01 = cds.expo(lbd,r,t0,RPV01 = True)
       return N*(spread-self.c)*RPV01
    
    def piecewise_expo(self,maturities,intensities,r=0,t0=0):
       T,R,freq,N = self.T,self.R,self.freq,self.N
       cds = cds_spreads(T,R,freq)
       spread,RPV01 = cds.piecewise_expo(maturities,intensities,t0,RPV01 = True)
       return N*(spread-self.c)*RPV01
    
    def Weibull(self,lbd,gamma,r=0,t0=0):
       T,R,freq,N = self.T,self.R,self.freq,self.N
       cds = cds_spreads(T,R,freq)
       spread,RPV01 = cds.Weibull(lbd,gamma,r,t0,RPV01 = True)
       return N*(spread-self.c)*RPV01
    
    def incomplete_logistic(self,lbd,r=0,t0=0):
       T,R,freq,N = self.T,self.R,self.freq,self.N
       cds = cds_spreads(T,R,freq)
       spread,RPV01 = cds.Weibull(lbd,r,t0,RPV01 = True)
       return N*(spread-self.c)*RPV01
   
    def log_logistic(self,alpha,beta,r=0,t0=0):
       T,R,freq,N = self.T,self.R,self.freq,self.N
       cds = cds_spreads(T,R,freq)
       spread,RPV01 = cds.Weibull(alpha,beta,r,t0,RPV01 = True)
       return N*(spread-self.c)*RPV01
   
    def log_normal(self,lbd,gamma,r=0,t0=0):
       T,R,freq,N = self.T,self.R,self.freq,self.N
       cds = cds_spreads(T,R,freq)
       spread,RPV01 = cds.Weibull(lbd,gamma,r,t0,RPV01 = True)
       return N*(spread-self.c)*RPV01 
#%%
def test_cds_spread():
    T,R = 5,0.4
    cds = cds_spreads(T,R)
    cds2 = cds_spreads(T,R,freq = 2)
    print(cds.expo(0.03))
    print(cds.expo(0.03,t0=1))
    print(cds2.expo(0.03))
    print(cds2.expo(0.03,t0=1))
    print(cds.piece_expo([2,5],[0.3,0.2,0.16]))
    print(cds.piece_expo([2,5],[0.2]*3))
    print(cds.piece_expo([2,5],[0.3,0.2,0.16],t0=1))
    cds_p = cds_values(T,R,0.018,1000)
    print(cds_p.expo(0.03))
    print(cds.expo(0.07,RPV01 = True))
    print(cds_p.expo(0.07))
    #print(cds.log_normal(0.1,0.6))
#test_cds_spread()
#%%
    