# -*- coding: utf-8 -*-

import numpy as np
from data import generator 
import matplotlib.pyplot as plt

A = generator.A
b = generator.b
coefs = generator.coefs

###########################Functions from problem
def F(x,lam, order = 1, A=A,b=b):
    #given function to minimize (full)
    F = 0
    for i in range(0,len(b)):
        F = F+1/2/len(b)*np.log(1+np.exp(-b[i]*np.dot(A[i],x)))
    F = F + lam*np.linalg.norm(x, ord=order)
    return F

def f(x, A=A, b= b):
    #given function (differentiable part)
    f= 0
    for i in range(0,len(b)):
        f = f+1/2/len(b)*np.log(1+np.exp(-b[i]*np.dot(A[i],x)))
    return f

def gradf(x, A=A, b= b):
    #given function gradient (differentiable part)
    F = 0
    for i in range(0,len(b)):
        F = F-1/2/len(b)*np.exp(-b[i]*np.dot(A[i],x))*A[i]*b[i]\
            /(1+np.exp(-b[i]*np.dot(A[i],x)))
    return F

def sgF(x,lam, A=A,b=b):
    #subgradient of given function
    F = 0
    for i in range(0,len(b)):
        F = F-1/2/len(b)*np.exp(-b[i]*np.dot(A[i],x))*A[i]*b[i]\
            /(1+np.exp(-b[i]*np.dot(A[i],x)))
    F = F + lam*((x<0)*(-1)+(x>0)*(1)+(x==0)*(0))
    return F

def optimal_step(x,sg,fun,coefs = coefs):
    #function to return the optimal step size w/ known soln
    #inputs: 
        #x: current x value
        #coefs: known optimal solution
        #sg: subgradient
        #F: function
    #ouputs:
        #eta: optimal step size
    return (fun(x)-fun(coefs))/(np.linalg.norm(sg(x),ord=2))**2

    

def apply_sub_grad(initial,step_size=optimal_step,sg=sgF, coefs=coefs, F = F, f= f, n_iter =1000):
    #Sub-gradient implementation
    #Inputs
        #initial: initial vec (x1)
        #step_size: function for step size rule
        #sg: subgradient of function
        #coefs: actual solution
    #Ouputs
        #sol: solution
        #errs: F(xt)-F(x*)
    errs = []
    x = initial
    i = 1
    sol = initial
    #while abs(F(x)-F(coefs))>1e-6:
    for l in range(0,n_iter):
        eta  = 1/np.sqrt(i)
        i += 1
        #eta = step_size(x,sg,f)
        y = x - eta*sg(x)
        errs.append(F(y)-F(coefs))
        if F(y)<F(x):
            sol = y
        x = y
        
    return errs,sol

def proxl1(x,lam):
    #proximal gradient operator for l1norm
    #inputs:
        #x: argument to evaluate at (vector)
        #lam: lambda parameter for soft-thresholding
    #ouputs:
        #\psi_st(x;\lam)
     return (x+lam)*(x<=-lam)+(x-lam)*(x>=lam)

def apply_prox_grad(initial, F, reg = proxl1, step_size = 1/2, grad = gradf, f =f, n_iter = 1000 ):
    #proximal gradient implementation for composite functions
    #inputs:
        #initial: initial vec
        #reg: regularization term for prox op
        #step_size: step size
        #grad: gradient of differentiable part
        #f: differentiable part of composite function
    #Ouputs
        #sol: solution
        #errs: F(xt)-F(x*)
    errs = []
    x = initial
    sol = initial
    i = 1
    #while abs(F(x)-F(coefs))>1e-8:
    for l in range(0,n_iter):
        eta = 1/np.sqrt(i)
        i+=1
        y = reg(x - eta*grad(x))
        i+=1
        errs.append(F(y)-F(coefs))
        print(F(y)-F(coefs))
        if F(y)<F(x):
            sol = y
        x = y
        
    return errs,sol
######################Applying all methods 
initial = np.random.random(len(coefs))*max(abs(coefs))
(errs,sol)  = apply_sub_grad(initial, F = lambda x :F(x, lam=.001), sg= lambda x :sgF(x, lam=.001), f =f)
(errsprox, solprox) = apply_prox_grad(initial,  F = lambda x :F(x, lam=.001),   reg = lambda x: proxl1(x,lam = .001))


########################Plotting
x = np.array(range(1,len(errs)+1))
plt.plot(x, errs)
plt.plot(x, errsprox)
plt.xlabel("t")
plt.ylabel("F(x^t)-F(x*)")
plt.legend(["Subgrad", "Proxgrad"])
plt.show()