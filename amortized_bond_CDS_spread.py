# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 06:26:29 2020

@author: Theodor
"""
class amortized_bond:
    
    def __init__(self,FV,c,T,freq,B=0):
        r"""
        FV: face value of the bond
        
        c =  the interest rate
        
        T = lifetime of the bond
        
        freq = frequency of payment
        
        B = baloon payment, if any. 
        
        """
        self.FV = FV
        self.c = c
        self.T = T
        self.freq = freq
        self.B = B
    
    def bond_annuity(self):
        FV,c,T,freq,B = self.FV,self.c,self.T,self.freq,self.B
        return bond_annuity(FV,c,T,freq,B)
            
    
    def bond_cash_flows(self):
        T,freq,B = self.T,self.freq,self.B
        return [self.bond_annuity()] * int(T*freq) if B==0 else \
               [self.bond_annuity() if i < (T*freq-1) else self.bond_annuity()+B \
                for i in range(T*freq)]
    
    def bond_value(self,r=0):
        T,freq = self.T,self.freq
        discounts = [1/(1+r/freq)**i for i in range(1,T*freq+1)]
        import numpy as np
        return np.dot(self.bond_cash_flows(),discounts)
      
    def principal_repayments(self):
        FV,c,T,freq,B = self.FV,self.c,self.T,self.freq,self.B
        k = int(T*freq)
        A = bond_annuity(FV,c,T,freq,B)
        return [A*(1+c/freq)**(i-1)-c/freq*(1+c/freq)**(i-1)*FV + B*(i==k) \
                for i in range(1,k+1)]
               
    def outstanding_notionals(self):
        FV,T,freq= self.FV,self.T,self.freq
        k = int(T*freq)
        return  [FV - sum(self.principal_repayments()[0:i]) for i in range(1,k+1)] 
    
    
def bond_annuity(FV,c,T,freq = 1,B = 0):
    r"""
    
    T: int
    
    FV: float
    
    c: float
    """
    return FV*c/freq* 1/(1-(1+c/freq)**(-T*freq)) if B == 0 else \
           bond_annuity(FV-B/(1+c/freq)**(T*freq),c,T,freq)

def outstanding_notional(FV,c,T,freq = 1,B = 0):
    r"""
    Returns function as output. 
    """
    am_bnd = amortized_bond(FV,c,T,freq,B)
    ons = am_bnd.outstanding_notionals()
    ons.insert(0,FV)
    times = [1/freq * i for i in range(0,T*freq+1)]
    return lambda x:sum([ons[i]*(times[i]<=x and x<times[i+1])  \
                         for i in range(len(times)-1)]+ [ons[-1]*(x>=ons[-1])])
            
#%%
class amortising_CDS:
    def __init__(self,FV,c,T,R,freq = 1,B = 0):
        r"""
        FV = face value (notional) of the credit hedged
        
        c = coupon of the hedged bond (To not be confused with the CDS spread)
        
        T = bond lifetime
        
        R = recovery rate
        
        """
        self.FV = FV
        self.c = c
        self.T = T
        self.freq = freq
        self.B = B
        self.R = R
    
    def expo(self,lbd,r = 0,t0 = 0):
        import numpy as np
        f = lambda t:np.exp(-lbd*t)*lbd
        S = lambda t:np.exp(-lbd*t)
        D = lambda t:np.exp(-r*t)
        FV,c,T,R,freq,B = self.FV,self.c,self.T,self.R,self.freq,self.B
        N = outstanding_notional(FV,c,T,freq,B)
        import scipy.integrate as integ
        q1 = integ.quad(lambda s:f(s)*D(s-t0)/S(t0)*(s>t0)*N(s),t0,T)[0]
        if freq==0:
            q2 = integ.quad(lambda s:S(s)*D(s-t0)/S(t0)*(s>t0)*N(s),t0,T)[0]
        else:
            dt = 1/freq
            times = [x for x in np.linspace(1/freq,T,int(T*freq)) if x>t0]
            q2 = dt * sum([S(x)/S(t0)*np.exp(-r*(x-t0)) * N(x) for x in times])
        return (1-R)*q1/q2 
    
    def piece_expo(self,maturities,intensities,r=0,t0=0,RPV01 = False):
        import numpy as np
        from piecewise_expo import piecewise_exponential
        FV,c,T,R,freq,B = self.FV,self.c,self.T,self.R,self.freq,self.B
        pe = piecewise_exponential(maturities,intensities)
        N = outstanding_notional(FV,c,T,freq,B)
        f = lambda t:pe.pdf2(t)
        S = lambda t:pe.survival2(t)
        D = lambda t:np.exp(-r*t)
        import scipy.integrate as integ
        q1 = integ.quad(lambda s:f(s)*D(s-t0)/S(t0)*(s>t0)*N(s),t0,T)[0]
        if freq==0:
            q2 = integ.quad(lambda s:S(s)*D(s-t0)/S(t0)*(s>t0)*N(s),t0,T)[0]
        else:
            dt = 1/freq
            times = [x for x in np.linspace(1/freq,T,int(T*freq)) if x>t0]
            q2 = dt * sum([S(x)/S(t0)*np.exp(-r*(x-t0)) * N(x) for x in times])
        return (1-R)*q1/q2 
    
    def Weibull(self,lbd,gamma,r=0,t0=0,RPV01 = False):
        import numpy as np
        f= lambda t:lbd*gamma*t**(gamma-1)*np.exp(-lbd*t**gamma)
        S = lambda t:np.exp(-lbd*t**gamma)
        D = lambda t:np.exp(-r*t)
        FV,c,T,R,freq,B = self.FV,self.c,self.T,self.R,self.freq,self.B
        N = outstanding_notional(FV,c,T,freq,B)
        import scipy.integrate as integ
        q1 = integ.quad(lambda s:f(s)*D(s-t0)/S(t0)*(s>t0)*N(s),t0,T)[0]
        if freq==0:
            q2 = integ.quad(lambda s:S(s)*D(s-t0)/S(t0)*(s>t0)*N(s),t0,T)[0]
        else:
            dt = 1/freq
            times = [x for x in np.linspace(1/freq,T,int(T*freq)) if x>t0]
            q2 = dt * sum([S(x)/S(t0)*np.exp(-r*(x-t0)) * N(x) for x in times])
        return (1-R)*q1/q2 
    
    def piecewise_Weibull(self,maturities,intensities,gamma,r=0,t0=0,RPV01 = False):
        import numpy as np
        from piecewise_weibull import piecewise_Weibull
        pw = piecewise_Weibull(maturities,intensities,gamma)
        f= lambda t:pw.pdf(t)
        S = lambda t:pw.survival(t)
        D = lambda t:np.exp(-r*t)
        FV,c,T,R,freq,B = self.FV,self.c,self.T,self.R,self.freq,self.B
        N = outstanding_notional(FV,c,T,freq,B)
        import scipy.integrate as integ
        q1 = integ.quad(lambda s:f(s)*D(s-t0)/S(t0)*(s>t0)*N(s),t0,T)[0]
        if freq==0:
            q2 = integ.quad(lambda s:S(s)*D(s-t0)/S(t0)*(s>t0)*N(s),t0,T)[0]
        else:
            dt = 1/freq
            times = [x for x in np.linspace(1/freq,T,int(T*freq)) if x>t0]
            q2 = dt * sum([S(x)/S(t0)*np.exp(-r*(x-t0)) * N(x) for x in times])
        return (1-R)*q1/q2 
    
    def Gompertz(self,eta,b,r=0,t0=0,RPV01 = False):
        import numpy as np
        f= lambda t:b*eta*np.exp(eta+b*t-eta*np.exp(b*t))
        S = lambda t:np.exp(-eta*(np.exp(b*t)-1))
        D = lambda t:np.exp(-r*t)
        FV,c,T,R,freq,B = self.FV,self.c,self.T,self.R,self.freq,self.B
        N = outstanding_notional(FV,c,T,freq,B)
        import scipy.integrate as integ
        q1 = integ.quad(lambda s:f(s)*D(s-t0)/S(t0)*(s>t0)*N(s),t0,T)[0]
        if freq==0:
            q2 = integ.quad(lambda s:S(s)*D(s-t0)/S(t0)*(s>t0)*N(s),t0,T)[0]
        else:
            dt = 1/freq
            times = [x for x in np.linspace(1/freq,T,int(T*freq)) if x>t0]
            q2 = dt * sum([S(x)/S(t0)*np.exp(-r*(x-t0)) * N(x) for x in times])
        return (1-R)*q1/q2 
    
    def expo_defer(self,lbd,r = 0,t0 = 0,T_reg_pmt=None,Tp = None):
        r"""
        T_reg_pmt = time of the last regular payment 
        
        Tp = time of the last payment
        """
        import numpy as np
        f = lambda t:np.exp(-lbd*t)*lbd
        S = lambda t:np.exp(-lbd*t)
        D = lambda t:np.exp(-r*t)
        FV,c,T,R,freq,B = self.FV,self.c,self.T,self.R,self.freq,self.B
        N = outstanding_notional(FV,c,T,freq,B)
        import scipy.integrate as integ
        q1 = integ.quad(lambda s:f(s)*D(s-t0)/S(t0)*(s>t0)*N(s),t0,T)[0]
        if freq==0:
            q2 = integ.quad(lambda s:S(s)*D(s-t0)/S(t0)*(s>t0)*N(s),t0,T)[0] \
                 if T_reg_pmt==None and Tp == None else \
                 integ.quad(lambda s:S(s)*D(s-t0)/S(t0)*(s>t0)*N(s),t0,T_reg_pmt)[0]\
                 +S(Tp)/S(t0)*np.exp(-r*(Tp-t0))
        else:
            dt = 1/freq
            times = [x for x in np.linspace(1/freq,T_reg_pmt,\
                    int(T_reg_pmt*freq)) if x>t0]+[Tp] if Tp!=None else \
                    [x for x in np.linspace(1/freq,T_reg_pmt,int(T_reg_pmt*freq))\
                     if x>t0] 
            q2 = dt * sum([S(x)/S(t0)*np.exp(-r*(x-t0)) * N(x) for x in times])
        return (1-R)*q1/q2
    
    def piece_expo_defer(self,maturities,intensities,r=0,t0=0,RPV01 = False\
                         ,T_reg_pmt=None,Tp = None):
        import numpy as np
        from piecewise_expo import piecewise_exponential
        FV,c,T,R,freq,B = self.FV,self.c,self.T,self.R,self.freq,self.B
        pe = piecewise_exponential(maturities,intensities)
        N = outstanding_notional(FV,c,T,freq,B)
        f = lambda t:pe.pdf2(t)
        S = lambda t:pe.survival2(t)
        D = lambda t:np.exp(-r*t)
        import scipy.integrate as integ
        q1 = integ.quad(lambda s:f(s)*D(s-t0)/S(t0)*(s>t0)*N(s),t0,T)[0]
        if freq==0:
            q2 = integ.quad(lambda s:S(s)*D(s-t0)/S(t0)*(s>t0)*N(s),t0,T)[0] \
                 if T_reg_pmt==None and Tp == None else \
                 integ.quad(lambda s:S(s)*D(s-t0)/S(t0)*(s>t0)*N(s),t0,T_reg_pmt)[0]\
                 +S(Tp)/S(t0)*np.exp(-r*(Tp-t0))
        else:
            dt = 1/freq
            times = [x for x in np.linspace(1/freq,T_reg_pmt,\
                    int(T_reg_pmt*freq)) if x>t0]+[Tp] if Tp!=None else \
                    [x for x in np.linspace(1/freq,T_reg_pmt,int(T_reg_pmt*freq))\
                     if x>t0] 
            q2 = dt * sum([S(x)/S(t0)*np.exp(-r*(x-t0)) * N(x) for x in times])
        return (1-R)*q1/q2
    
    def Weibull_defer(self,lbd,gamma,r=0,t0=0,RPV01 = False,\
                      T_reg_pmt=None,Tp = None):
        import numpy as np
        f= lambda t:lbd*gamma*t**(gamma-1)*np.exp(-lbd*t**gamma)
        S = lambda t:np.exp(-lbd*t**gamma)
        D = lambda t:np.exp(-r*t)
        FV,c,T,R,freq,B = self.FV,self.c,self.T,self.R,self.freq,self.B
        N = outstanding_notional(FV,c,T,freq,B)
        import scipy.integrate as integ
        q1 = integ.quad(lambda s:f(s)*D(s-t0)/S(t0)*(s>t0)*N(s),t0,T)[0]
        if freq==0:
            q2 = integ.quad(lambda s:S(s)*D(s-t0)/S(t0)*(s>t0)*N(s),t0,T)[0] \
                 if T_reg_pmt==None and Tp == None else \
                 integ.quad(lambda s:S(s)*D(s-t0)/S(t0)*(s>t0)*N(s),t0,T_reg_pmt)[0]\
                 +S(Tp)/S(t0)*np.exp(-r*(Tp-t0))
        else:
            dt = 1/freq
            times = [x for x in np.linspace(1/freq,T_reg_pmt,\
                    int(T_reg_pmt*freq)) if x>t0]+[Tp] if Tp!=None else \
                    [x for x in np.linspace(1/freq,T_reg_pmt,int(T_reg_pmt*freq))\
                     if x>t0] 
            q2 = dt * sum([S(x)/S(t0)*np.exp(-r*(x-t0)) * N(x) for x in times])
        return (1-R)*q1/q2
    
    def piecewise_Weibull_defer(self,maturities,intensities,gamma,\
                                r=0,t0=0,RPV01 = False,T_reg_pmt=None,Tp = None):
        import numpy as np
        from piecewise_weibull import piecewise_Weibull
        pw = piecewise_Weibull(maturities,intensities,gamma)
        f= lambda t:pw.pdf(t)
        S = lambda t:pw.survival(t)
        D = lambda t:np.exp(-r*t)
        FV,c,T,R,freq,B = self.FV,self.c,self.T,self.R,self.freq,self.B
        N = outstanding_notional(FV,c,T,freq,B)
        import scipy.integrate as integ
        q1 = integ.quad(lambda s:f(s)*D(s-t0)/S(t0)*(s>t0)*N(s),t0,T)[0]
        if freq==0:
            q2 = integ.quad(lambda s:S(s)*D(s-t0)/S(t0)*(s>t0)*N(s),t0,T)[0] \
                 if T_reg_pmt==None and Tp == None else \
                 integ.quad(lambda s:S(s)*D(s-t0)/S(t0)*(s>t0)*N(s),t0,T_reg_pmt)[0]\
                 +S(Tp)/S(t0)*np.exp(-r*(Tp-t0))
        else:
            dt = 1/freq
            times = [x for x in np.linspace(1/freq,T_reg_pmt,\
                    int(T_reg_pmt*freq)) if x>t0]+[Tp] if Tp!=None else \
                    [x for x in np.linspace(1/freq,T_reg_pmt,int(T_reg_pmt*freq))\
                     if x>t0] 
            q2 = dt * sum([S(x)/S(t0)*np.exp(-r*(x-t0)) * N(x) for x in times])
        return (1-R)*q1/q2
    
    def Gompertz_defer(self,eta,b,r=0,t0=0,RPV01 = False\
                           ,T_reg_pmt=None,Tp = None):
        import numpy as np
        f= lambda t:b*eta*np.exp(eta+b*t-eta*np.exp(b*t))
        S = lambda t:np.exp(-eta*(np.exp(b*t)-1))
        D = lambda t:np.exp(-r*t)
        FV,c,T,R,freq,B = self.FV,self.c,self.T,self.R,self.freq,self.B
        N = outstanding_notional(FV,c,T,freq,B)
        import scipy.integrate as integ
        q1 = integ.quad(lambda s:f(s)*D(s-t0)/S(t0)*(s>t0)*N(s),t0,T)[0]
        if freq==0:
            q2 = integ.quad(lambda s:S(s)*D(s-t0)/S(t0)*(s>t0)*N(s),t0,T)[0] \
                 if T_reg_pmt==None and Tp == None else \
                 integ.quad(lambda s:S(s)*D(s-t0)/S(t0)*(s>t0)*N(s),t0,T_reg_pmt)[0]\
                 +S(Tp)/S(t0)*np.exp(-r*(Tp-t0))
        else:
            dt = 1/freq
            times = [x for x in np.linspace(1/freq,T_reg_pmt,\
                    int(T_reg_pmt*freq)) if x>t0]+[Tp] if Tp!=None else \
                    [x for x in np.linspace(1/freq,T_reg_pmt,int(T_reg_pmt*freq))\
                     if x>t0] 
            q2 = dt * sum([S(x)/S(t0)*np.exp(-r*(x-t0)) * N(x) for x in times])
        return (1-R)*q1/q2
    
#%%