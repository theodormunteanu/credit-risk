# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 12:19:01 2020

@author: Theodor
"""
"""
Markov chain risky bond prices
All 5 examples have probability input = P. 
Test 1: 5Y bond with coupon = 8 % and notional = 100,000
Test 1 bis: a 3Y bond with notional = 50,000. 

Test 2: value of a 3-year rating dependent bond with coupons depending  as 
        follows: $c(R(t)) = \begin{cases}5\%, R(t)=R_1,\\
                                        7\%, R(t) = R_2, \\
                                        9\%, R(t) = R_3 \\
                            \end{cases}$
Test 3: value of a 3-year rating dependent bond with coupons and notional dep

Test 4: amortizing bond.                              
"""
#%%
def test_price_bond1():
    r"""
    Bond price from example 2 per 100 par value. 
    
    """
    from scipy.linalg import logm,expm
    import numpy as np
    P = np.array([[0.94,0.03,0.02,0.01],[0.1,0.8,0.07,0.03],\
                  [0.05,0.15,0.6,0.2],[0,0,0,1]])
    lbd = logm(P)
    e3,e4 = np.array([0,0,1,0]).T,np.array([0,0,0,1]).T
    def S(u,t = 0):
        "conditional survival function"
        if t==0:
            return 1-np.dot(np.dot(e3.T,expm(u*lbd)),e4)
        else:
            return S(u)/S(t)
    def f(u,t = 0):
        "conditional density function"
        if t==0:
            return np.dot(np.dot(e3.T,lbd),np.dot(expm(u*lbd),e4))
        else:
            return np.dot(np.dot(e3.T,lbd),np.dot(expm(u*lbd),e4))/(1-S(t))
    price0 = 8*(S(1)+S(2)+S(3)+S(4)+S(5))+100*S(5)+50*np.dot(np.dot(e3.T,\
               expm(5*lbd)-np.identity(4)),e4)
    price1 = 8*(S(1,7/12)+S(2,7/12)+S(3,7/12)+S(4,7/12)+S(5,7/12))+100*S(5,7/12)+\
             +50*np.dot(np.dot(e3.T,expm(53/12 * lbd)-np.identity(4)),e4)
    print(price0,price1,price1-price0)
test_price_bond1()
#%%
def test_price_bond1bis():
    from scipy.linalg import logm,expm
    import numpy as np
    P = np.array([[0.94,0.03,0.02,0.01],[0.1,0.8,0.07,0.03],\
                  [0.05,0.15,0.6,0.2],[0,0,0,1]])
    lbd = logm(P)
    e3,e4 = np.array([0,0,1,0]).T,np.array([0,0,0,1]).T
    def S(u,t = 0):
        "conditional survival function"
        if t==0:
            return 1-np.dot(np.dot(e3.T,expm(u*lbd)),e4)
        else:
            return S(u)/S(t)
    def f(u,t = 0):
        "conditional density function"
        if t==0:
            return np.dot(np.dot(e3.T,lbd),np.dot(expm(u*lbd),e4))
        else:
            return np.dot(np.dot(e3.T,lbd),np.dot(expm(u*lbd),e4))/(1-S(t))
    price0 = 4*(S(1)+S(2)+S(3))+50*S(3)+25*np.dot(np.dot(e3.T,\
               expm(3*lbd)-np.identity(4)),e4)
    price1 = 4*(S(1,7/12)+S(2,7/12)+S(3,7/12))+50*S(3,7/12)+\
             +25*np.dot(np.dot(e3.T,expm(29/12 * lbd)-np.identity(4)),e4)
    print(price0,price1,price1-price0)
test_price_bond1bis()
#%%
def test_price_bond2():
    r"""
    Example 3 from here. 
    """
    from scipy.linalg import logm,expm
    import numpy.linalg as la
    import numpy as np
    P = np.array([[0.94,0.03,0.02,0.01],[0.1,0.8,0.07,0.03],\
                  [0.05,0.15,0.6,0.2],[0,0,0,1]])
    P1,P2,P3 = la.matrix_power(P,1),la.matrix_power(P,2),la.matrix_power(P,3)
    e3,e4 = np.array([0,0,1,0]).T,np.array([0,0,0,1]).T
    FV = 50000
    #avg_c1 is the average coupon at time 1
    avg_c1 = FV*(0.05*P1[2][0]/(P1[2][0]+P1[2][1]+P1[2][2])+0.07* \
                  P1[2][1]/(P1[2][0]+P1[2][1]+P1[2][2]) + \
                  0.09 * P1[2][2]/(P1[2][0]+P1[2][1]+P1[2][2]))
    avg_c2 = FV*(0.05*P2[2][0]/(P2[2][0]+P2[2][1]+P2[2][2])+0.07* \
                  P2[2][1]/(P2[2][0]+P2[2][1]+P2[2][2]) + \
                  0.09 * P2[2][2]/(P2[2][0]+P2[2][1]+P2[2][2]))
    avg_c3 = FV*(0.05*P3[2][0]/(P3[2][0]+P3[2][1]+P3[2][2])+0.07* \
                  P3[2][1]/(P3[2][0]+P3[2][1]+P3[2][2]) + \
                  0.09 * P3[2][2]/(P3[2][0]+P3[2][1]+P3[2][2]))
    lbd = logm(P)
    #q1 is the notional received in case there is no default
    #q2  is the notional received in case there is default
    q1 = FV*(1-np.dot(e3.T,np.dot(P3,e4)))
    q2 = FV * 1/2 * np.dot(e3.T,np.dot(expm(3*lbd)-np.identity(4),e4))
    print(avg_c1+avg_c2+avg_c3 + q1 + q2)
test_price_bond2()
#%%
def test_price_bond3():
    from scipy.linalg import logm,expm
    import numpy.linalg as la
    import numpy as np
    P = np.array([[0.94,0.03,0.02,0.01],[0.1,0.8,0.07,0.03],\
                  [0.05,0.15,0.6,0.2],[0,0,0,1]])
    P1,P2,P3 = la.matrix_power(P,1),la.matrix_power(P,2),la.matrix_power(P,3)
    e3,e4 = np.array([0,0,1,0]).T,np.array([0,0,0,1]).T
    FV,R = 50000,0.5
    avg_c1 = FV*(0.05*P1[2][0]/(P1[2][0]+P1[2][1]+P1[2][2])+0.07* \
                  P1[2][1]/(P1[2][0]+P1[2][1]+P1[2][2]) + \
                  0.09 * P1[2][2]/(P1[2][0]+P1[2][1]+P1[2][2]))
    avg_c2 = FV*(0.05*P2[2][0]/(P2[2][0]+P2[2][1]+P2[2][2])+0.07* \
                  P2[2][1]/(P2[2][0]+P2[2][1]+P2[2][2]) + \
                  0.09 * P2[2][2]/(P2[2][0]+P2[2][1]+P2[2][2]))
    avg_c3 = FV*(0.05*P3[2][0]/(P3[2][0]+P3[2][1]+P3[2][2])+0.07* \
                  P3[2][1]/(P3[2][0]+P3[2][1]+P3[2][2]) + \
                  0.09 * P3[2][2]/(P3[2][0]+P3[2][1]+P3[2][2]))
    avg_c4 = FV*(P3[2][0]+1.05*P3[2][1] + 1.1 * P3[2][2]+(1-R)*P3[2][3])
    lbd = logm(P)
    q2 = FV * 1/2 * np.dot(e3.T,np.dot(expm(3*lbd)-np.identity(4),e4))
    print(avg_c1+avg_c2+avg_c3+avg_c4)
    
test_price_bond3()
#%%
def test_price_bond4():
    from scipy.linalg import logm,expm
    import numpy.linalg as la
    import numpy as np
    P = np.array([[0.94,0.03,0.02,0.01],[0.1,0.8,0.07,0.03],\
                  [0.05,0.15,0.6,0.2],[0,0,0,1]])
    P1,P2,P3 = la.matrix_power(P,1),la.matrix_power(P,2),la.matrix_power(P,3)
    FV,R,r = 50000,0.5,0.02
    A = FV*(1-np.exp(-r))/(np.exp(-r)-np.exp(-4*r))
    print(A)
    notional = lambda u: 3*A * (0<=u  and u<=1)+2*A * (1<u and u<=2) + A*(2<u and u<=3)
    e3,e4 = np.array([0,0,1,0]).T,np.array([0,0,0,1]).T
    lbd = logm(P)
    f = lambda u: notional(u)*np.exp(-r * u)*(np.dot(np.dot(e3.T, expm(u *lbd)),e4))
    import scipy.integrate as integ
    q1 = integ.quad(f,0,3)[0] 
    "q1: the present value of the recovered notional |default"
    pv_annuities = [A*(1-np.dot(np.dot(e3.T,la.matrix_power(P,1)),e4))*np.exp(-r),\
                    A*(1-np.dot(np.dot(e3.T,la.matrix_power(P,2)),e4))*np.exp(-2*r),\
                    A*(1-np.dot(np.dot(e3.T,la.matrix_power(P,3)),e4))*np.exp(-3*r)]
    print(pv_annuities)
    print(sum(pv_annuities)+q1)
test_price_bond4()