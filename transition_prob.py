# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 15:27:37 2020

@author: Theodor
"""

def test_tpm():
    r"""
    Transition probability matrices and spreads 
    
    """
    from scipy.linalg import logm,expm
    import numpy as np
    P2 = np.array([[0.94,0.03,0.02,0.01],[0.1,0.8,0.07,0.03],\
                  [0.05,0.15,0.6,0.2],[0,0,0,1]])
    P4 = np.array([[0.846,0.069,0.057,0.028],[0.146,0.638,0.112,0.104],\
                   [0.164,0.143,0.365,0.328],[0,0,0,1]])
    P6 = np.array([[0.8015,0.0833,0.0726,0.0426],[0.2613,0.542,0.1506,0.0461],\
                   [0.126,0.1832,0.5026,0.1982],[0,0,0,1]])
    e1,e2 = np.array([1,0,0,0]).T,np.array([0,1,0,0]).T
    e3,e4 = np.array([0,0,1,0]).T,np.array([0,0,0,1]).T
    "the 3 generators"
    lbd1,lbd2,lbd3 = logm(P2)/2,logm(P4)/4,logm(P6)/6
    R = 0.4
    "the piecewise exponential paramteters for each risk class: A,B and C"
    lbdA = [(1-np.log(1-np.dot(np.dot(e1.T,P2),e4)))/2,\
            (np.log(1-np.dot(np.dot(e1.T,P2),e4))-np.log(1-np.dot(np.dot(e1.T,P4),e4)))/2,\
            (np.log(1-np.dot(np.dot(e1.T,P4),e4))-np.log(1-np.dot(np.dot(e1.T,P6),e4)))/2]
    lbdB = [(1-np.log(1-np.dot(np.dot(e2.T,P2),e4)))/2,\
            (np.log(1-np.dot(np.dot(e2.T,P2),e4))-np.log(1-np.dot(np.dot(e2.T,P4),e4)))/2,\
            (np.log(1-np.dot(np.dot(e2.T,P4),e4))-np.log(1-np.dot(np.dot(e2.T,P6),e4)))/2]
    lbdC = [(1-np.log(1-np.dot(np.dot(e3.T,P2),e4)))/2,\
            (np.log(1-np.dot(np.dot(e3.T,P2),e4))-np.log(1-np.dot(np.dot(e3.T,P4),e4)))/2,\
            (np.log(1-np.dot(np.dot(e3.T,P4),e4))-np.log(1-np.dot(np.dot(e3.T,P6),e4)))/2]
    Lbd = lambda u: lbd1*u*(0<=u and u<=2)+(lbd1*2 + lbd2 * (u-2))*(2<u and u<=4)+\
                    +(lbd1*2+lbd2*2+lbd3*(u-2))*(4<u and u<=6)
    survivalsA = [1-np.dot(np.dot(e1.T,expm(Lbd(x))),e4) for x in [1,2,3,4,5,6]]
    survivalsB = [1-np.dot(np.dot(e2.T,expm(Lbd(x))),e4) for x in [1,2,3,4,5,6]]
    survivalsC = [1-np.dot(np.dot(e3.T,expm(Lbd(x))),e4) for x in [1,2,3,4,5,6]]
    s0A = (1-R)*(1-np.exp(-2*lbdA[0]-2*lbdA[1]-2*lbdA[2]))/sum(survivalsA)
    s0B = (1-R)*(1-np.exp(-2*lbdB[0]-2*lbdB[1]-2*lbdB[2]))/sum(survivalsB)
    s0C = (1-R)*(1-np.exp(-2*lbdC[0]-2*lbdC[1]-2*lbdC[2]))/sum(survivalsC)
    print("The spread of risk class A for 6 years holding is",s0A)
    print("The spread of risk class B for 6 years holding is",s0B)
    print("The spread of risk class C for 6 years holding is",s0C)
    survivalsA_biann = [1-np.dot(np.dot(e1.T,expm(Lbd(x))),e4) for x in np.linspace(1/2,6,12)]
    survivalsB_biann = [1-np.dot(np.dot(e2.T,expm(Lbd(x))),e4) for x in np.linspace(1/2,6,12)]
    survivalsC_biann = [1-np.dot(np.dot(e3.T,expm(Lbd(x))),e4) for x in np.linspace(1/2,6,12)]
    s0A_biann = 2*(1-R)*(1-np.exp(-2*lbdA[0]-2*lbdA[1]-2*lbdA[2]))/sum(survivalsA_biann)
    s0B_biann = 2*(1-R)*(1-np.exp(-2*lbdB[0]-2*lbdB[1]-2*lbdB[2]))/sum(survivalsB_biann)
    s0C_biann = 2*(1-R)*(1-np.exp(-2*lbdC[0]-2*lbdC[1]-2*lbdC[2]))/sum(survivalsC_biann)
    print("The spread of risk class A for 6 years holding, semi-ann paym is",s0A_biann)
    print("The spread of risk class B for 6 years holding, semi-ann paym is",s0B_biann)
    print("The spread of risk class C for 6 years holding, semi-ann paym is",s0C_biann)
    survivalsA_quart = [1-np.dot(np.dot(e1.T,expm(Lbd(x))),e4) for x in np.linspace(1/4,6,24)]
    survivalsB_quart = [1-np.dot(np.dot(e2.T,expm(Lbd(x))),e4) for x in np.linspace(1/4,6,24)]
    survivalsC_quart = [1-np.dot(np.dot(e3.T,expm(Lbd(x))),e4) for x in np.linspace(1/4,6,24)]
    s0A_quart = 4*(1-R)*(1-np.exp(-2*lbdA[0]-2*lbdA[1]-2*lbdA[2]))/sum(survivalsA_quart)
    s0B_quart = 4*(1-R)*(1-np.exp(-2*lbdB[0]-2*lbdB[1]-2*lbdB[2]))/sum(survivalsB_quart)
    s0C_quart = 4*(1-R)*(1-np.exp(-2*lbdC[0]-2*lbdC[1]-2*lbdC[2]))/sum(survivalsC_quart)
    print("The spread of risk class A for 6 years holding, quarterly paym is",s0A_quart)
    print("The spread of risk class B for 6 years holding, quarterly paym is",s0B_quart)
    print("The spread of risk class C for 6 years holding, quarterly paym is",s0C_quart)
test_tpm()
#%%