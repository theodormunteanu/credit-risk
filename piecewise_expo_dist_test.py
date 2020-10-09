# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 14:17:56 2020

@author: Theodor
"""

#%%
def test_pds():
    rand_var1 = piecewise_exponential([1/2,1,3],[0.0216,0.0433,0.04375,0.0941])
    rand_var2 = piecewise_exponential([3/4,2,4],[0.0216,0.0625,0.0225,0.1658])
    pds1 = [1-rand_var1.survival(i+1)/rand_var1.survival(i) for i in range(0,5)]
    pds2 = [1-rand_var2.survival(i+1)/rand_var2.survival(i) for i in range(0,5)]
    print(pds1)
    print(pds2)
test_pds()
#%%