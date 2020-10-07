# -*- coding: utf-8 -*-
"""
Created on May 15, 2020

@author: Rohan Chakraborty

SEIRSD model based on the articles by Henri Froese
S: Susceptibles
E: Exposed
I: Infected
R: Recovered
D: Dead

with lost of immunity of those who recovered after 1 year.
In addition case-dependent fatality rate and time-dependent basic
reproduction number R0.

"""
from scipy.integrate import odeint
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# function that returns derivative 
def model(y,t,alpha_0, beta_f_start, gamma, delta, no_immu, rho, R0_start, N):

    y1=y[0]
    y2=y[1]
    y3=y[2]
    y4=y[3]
    y5=y[4]

    
    # time-dependent basic reproduction rate R0=R0(t)

    R0_end=1 #R0_start/2;
    k=0.5
    t_0=1000
    #R0=(R0_start-R0_end)/(1+exp(-k*(t_0-t)))+R0_end;
    #R0=(1.2*R0_start*cos(t/400))^2;

    # time-dependent and resource-dependent and age-group weighted fatality rate
    aI=0.0001;aII=0.005;aIII=0.05;aIV=0.2 # age-dependent fatality rate
    pI=0.1;pII=0.3;pIII=0.4;pIV=0.2 # distribution

    alpha_avg=aI*pI+aII*pII+aIII*pIII+aIV*pIV

    # max number of beds in intensive care
    beds=50000

    # about 20% of the infected need intensive care
    patients=y3*0.2

    if patients>beds:
        R0=R0_start #/2
        s=0.5
        alpha=alpha_avg+s*y3/N
    else:
        R0=R0_start
        alpha=alpha_avg


    beta_f=R0*gamma
    beta=beta_f/N

    
    # model    
    dy1dt = -beta*y1*y3+no_immu*y4
    dy2dt = beta*y1*y3-delta*y2
    dy3dt = delta*y2- (1-alpha)*gamma*y3-alpha*rho*y3
    dy4dt = (1-alpha)*gamma*y3-no_immu*y4
    dy5dt = alpha*rho*y3
    
    return [dy1dt, dy2dt, dy3dt, dy4dt, dy5dt]



def main():
    print("SEIRSD model")

    N=80e06; # total population
    I0=10
    # initial condition
    y0 = [N-I0,0, I0,0,0] # 10 infected, no immune, no recovered, no dead

    # parameters
    
    # recovery rate
    gamma=1/20 # duration of illness is 20 days,
    # i.e. an infected can infect susceptibles over an period of 20 days

    no_immu=1/365 # after 1 year loss of immunity

    # death rate
    rho=1/10 # or people die after 10 days on average

    # fatality rate
    alpha_0=0.5/100 # 0.5 % of the infected die

    # exposed: latent period is 7 days
    delta=1/7
    
    # transmission coefficient
        
    c=5/7 # contact rate: 7 contacts per week, assumed to be constant
    
    p=0.1 # infection probability per contact, assumed to be constant
    
    beta_f_start=c*p # transmission coefficient, 1/days
    
    # basic reproduction number R0
    R0_start=beta_f_start/gamma
    # beta_f=gamma*R0
    
    # beta_f=1.67 # 1/days
    
    #beta=beta_f/N
    
    # time points
    t = np.linspace(0,10*365,100000) # 10 years

    # solve ODE
    y = odeint(model,y0,t,args=(alpha_0, beta_f_start, gamma, delta, no_immu,rho,R0_start,N))

    y1=y[:,0]
    y2=y[:,1]
    y3=y[:,2]
    y4=y[:,3]
    y5=y[:,4]    

    total=y1+y2+y3+y4+y5
    
    # plot results

    # 3D-Plot
  
    #from mpl_toolkits.mplot3d import Axes3D
    #mpl.rcParams['legend.fontsize'] = 10
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #ax.plot(y1, y2, y3, label='orbit')
    #ax.legend()
    #plt.show()


    f, ax1 = plt.subplots(1,1,figsize=(10,4))
    ax1.plot(t, y1, 'b', alpha=0.7, linewidth=2, label='Susceptible')
    ax1.plot(t, y2, 'y', alpha=0.7, linewidth=2, label='Exposed')
    ax1.plot(t, y3, 'r', alpha=0.7, linewidth=2, label='Infected')
    ax1.plot(t, y4, 'g', alpha=0.7, linewidth=2, label='Recovered')
    ax1.plot(t, y5, 'k', alpha=0.7, linewidth=2, label='Dead')
    ax1.plot(t, y1+y2+y3+y4+y5, 'c--', alpha=0.7, linewidth=2, label='Total')
    ax1.set_xlabel('Time (days)')
    #ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax1.legend(borderpad=2.0)
    legend.get_frame().set_alpha(0.5)
    plt.title("SEIRSD model")
    plt.show()


    f, ax2 = plt.subplots(1,1,figsize=(10,4))
    ax2.semilogy(t, y1, 'b', alpha=0.7, linewidth=2, label='Susceptible')
    ax2.semilogy(t, y2, 'y', alpha=0.7, linewidth=2, label='Exposed')
    ax2.semilogy(t, y3, 'r', alpha=0.7, linewidth=2, label='Infected')
    ax2.semilogy(t, y4, 'g', alpha=0.7, linewidth=2, label='Recovered')
    ax2.semilogy(t, y5, 'k', alpha=0.7, linewidth=2, label='Dead')
    ax2.semilogy(t, y1+y2+y3+y4+y5, 'c--', alpha=0.7, linewidth=2, label='Total')
    ax2.set_xlabel('Time (days)')
    ax2.grid(True, which="both")
    plt.ylim([1,N*1.1])
    #ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax2.legend(borderpad=2.0)
    legend.get_frame().set_alpha(0.5)
    plt.title("SEIRSD model")
    plt.show()
    
if __name__ == '__main__':
    # call main program
    main()

    
    
