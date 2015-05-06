# MPC for crowds

file: `mpc_crowds_yalmip.m`

In this file we model the protocol as a discrete-time Markov chain. We analyze the problem in a model predictive control fashion with the following formulation

x(k+1) = A x(k) + B u(k)

This file required [YALMIP](http://users.isy.liu.se/johanl/yalmip/) with a quadratic solver to run, release R20150204 in [MATLAB](www.mathworks.com/products/matlab/) R2014a.

A report of this work can be found [here](https://github.com/marcotinacci/predictive-crowds/blob/master/documentation/main.pdf).