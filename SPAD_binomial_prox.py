#!/usr/bin/env python3
"""Exact convex proximal step for a static T-exposure Bernoulli SPAD count image.

Solves min_{0<=Y<=1} -k log p(Y) -(T-k) log(1-p(Y)) + rho/2*(Y-z)^2
where p(Y)=1-exp(-(B*Y+background_total)/T).

The array-level solver is vectorized and uses monotonic derivative bisection.
This is a building block for PnP/HQS, NOT a complete validated PnP method.
"""
import numpy as np


def negloglik_y(y,k,B,T=64,bg=.01):
    y=np.asarray(y,dtype=float)
    k=np.asarray(k,dtype=float)
    lam=(B*y+bg)/T
    p=np.clip(-np.expm1(-lam),1e-14,1-1e-14)
    return -k*np.log(p)-(T-k)*np.log1p(-p)


def prox_binomial(z,k,B,rho,T=64,bg=.01,iterations=48):
    if B<=0 or T<1 or rho<=0 or bg<0:
        raise ValueError('B, T, and rho must be positive, bg nonnegative')
    z=np.asarray(z,dtype=np.float64)
    k=np.asarray(k,dtype=np.float64)
    if np.any((k<0)|(k>T)):
        raise ValueError('k must be in [0,T]')
    z,k=np.broadcast_arrays(z,k)
    low=np.zeros(z.shape,dtype=np.float64)
    high=np.ones(z.shape,dtype=np.float64)
    def deriv(y):
        p=np.maximum(-np.expm1(-(B*y+bg)/T),1e-14)
        # df/dY = B/T*(T-k/p) + rho*(Y-z)
        return (B/T)*(T-k/p)+rho*(y-z)
    # Minimum is at boundary if derivative at boundary does not cross 0.
    at_lo=deriv(low)>=0
    at_hi=deriv(high)<=0
    for _ in range(iterations):
        mid=(low+high)*0.5
        downhill=deriv(mid)<0
        low=np.where(downhill,mid,low)
        high=np.where(downhill,high,mid)
    ans=(low+high)*0.5
    return np.where(at_lo,0.,np.where(at_hi,1.,ans))


def selftest():
    from scipy.optimize import minimize_scalar
    rng=np.random.default_rng(145)
    # Exhaustive probes near k=0 and k=T, near boundaries, and across budgets.
    max_diff=0.
    for B in [.5,1.,2.,4.,8.]:
      for rho in [.1,1.,10.,100.]:
       n=150
       zz=rng.uniform(-.2,1.2,size=n)
       kk=rng.integers(0,65,size=n)
       kk[:3]=[0,1,64]
       res=prox_binomial(zz,kk,B,rho)
       for j in range(n):
        fun=lambda y: float(negloglik_y(y,float(kk[j]),B)) + rho/2*(y-zz[j])**2
        brute=minimize_scalar(fun,method='bounded',bounds=(0,1),options={'xatol':1e-12})
        candidate=[(fun(0),0),(fun(1),1),(fun(brute.x),brute.x)]
        answer=min(candidate)[1]
        delta=abs(res[j]-answer)
        max_diff=max(max_diff,delta)
        if abs(fun(res[j])-fun(answer))>1e-7 or delta>2e-5:
            raise AssertionError((B,rho,kk[j],zz[j],res[j],answer))
    print('PASS: 3000 proximal subproblems against independent scipy bounded minimization.')
    print('Max absolute Y difference:',max_diff)

if __name__=='__main__':
    selftest()
