###################################
########Import Libraries###########
###################################
#Begin
from __future__ import division
from skmonaco import mcquad
from cmath import *
import scipy.integrate 
import numpy
import time
import numpy as np
import pylab as pl
from random import random
start_time = time.time()
#End
###################################

###################################
###########Fermi Function##########
###################################
#Begin
def fermi(e):  # e: dispersion energy, B: inverse temperature
  val = 1.0/(exp(beta*e)+1.0) 
  return val
#End
###################################

###################################
###########Bose Function##########
###################################
#Begin
def bose(e):  # e: dispersion energy, B: inverse temperature
  if e!=0:
    val = 1.0/(exp(beta*(e))-1.0) 
  else:          # if e==0 then use a regulator
    reg=0.01
    val = 1.0/(exp(beta*e-reg)-1.0) 
  return val
#End
###################################

###################################
############Dispersion#############
###################################
#Begin
def E(p): # p: momentum, m: chemical potential, t: hopping amplitude, a: lattice constant
  val = -2*t*(cos(p[0]*a)+cos(p[1]*a))+mu-SIGMA
  return val
#End
###################################

###################################
########Integrand Hartree##########
###################################
#Begin
def Integrand_1st_order_new(p): # p: momentum, m: chemical potential, t: hopping amplitude, a: lattice constant, B: inverse temperature, u: Hubbard potential
  dis_en=E(p)
  val=U*fermi(dis_en)/2/2/pi/pi
  return val
###################################

###################################
########Pick a random point########
###################################
#Begin
def pick(a,b):
  r1=random()
  new=r1*(b-a)+a
  return new 
#End
###################################

###################################
#########Naive MC Hartree##########
###################################
#Begin
def naive_mc_Hartree(dim,N):
  hits=0
  p=[None]*dim
  chain=[]
  for i in range (0,N):
    for j in range (0,dim):
      p[j]=pick(-pi,pi)  # Pick momentum
      
    hits=hits+Integrand_1st_order_new(p)
    chain.append(Integrand_1st_order_new(p))
  
  s=0
  avg=hits/N  
  measurement=((2*pi)**dim)*hits/N   
  for i in range (0,N):
    s=s+(chain[i]-avg)*(chain[i]-avg)
  s=s/N
  s=sqrt(s)
  uncertainty = s/sqrt(N)*(2*pi)**dim 
  return measurement, uncertainty
#End
###################################

############Integrand##############
###################################
#Begin
def Integrand_2nd_order(p): # p: momentum, m: chemical potential, t: hopping amplitude, a: lattice constant, B: inverse temperature, u: Hubbard potential
  k1 = [p[2],p[3]]
  e1=E(k1)
  k2=[p[0],p[1]]
  e2=E(k2)
  k3=[px_ext+p[2]-p[0],py_ext+p[3]-p[1]]
  e3=E(k3)
  val=fermi(e1)-fermi(-e3)
  val=val*(fermi(e2)+bose(e1+e3)) 
  val=-U*U*val/2/2/2/2/pi/pi/pi/pi  # Only the prefactor
  val=val/(1j*nu_ext-e1-e3+e2)
  return val
#End
###################################

###################################
#######Integrand real part#########
###################################
#Begin
def Int_2nd_Real(p):
  val = float( Integrand_2nd_order(p).real )
  return val
#End
###################################

###################################
#######Integrand imag part#########
###################################
#Begin
def Int_2nd_Imag(p):
  val = float( Integrand_2nd_order(p).imag )
  return val
#End
###################################

###################################
#############Naive MC##############
###################################
#Begin
def naive_mc_2nd(dim,N):
  hits_imag=0
  hits_real=0
  hits=0
  p=[None]*dim
  chain_imag=[]
  chain_real=[]
  for i in range (0,N):
    for j in range (0,dim):
      p[j]=pick(-pi,pi)  # Pick momentum
      
    hits_imag=hits_imag+Int_2nd_Imag(p)
    chain_imag.append(Int_2nd_Imag(p))
    hits_real=hits_real+Int_2nd_Real(p)
    chain_real.append(Int_2nd_Real(p))
    hits=hits+Integrand_2nd_order(p)
  
  s_imag=0
  avg_imag=hits_imag/N 
  print '****', ((2*pi)**dim)*hits/N 
  #print 'avg =', avg
  #print chain
  measurement_imag=((2*pi)**dim)*hits_imag/N   
  for i in range (0,N):
    s_imag=s_imag+(chain_imag[i]-avg_imag)*(chain_imag[i]-avg_imag)
  s_imag=s_imag/N
  s_imag=sqrt(s_imag)
  uncertainty_imag = s_imag/sqrt(N)*(2*pi)**dim 

  s_real=0
  avg_real=hits_real/N  
  #print 'avg =', avg
  #print chain
  measurement_real=((2*pi)**dim)*hits_real/N   
  for i in range (0,N):
    s_real=s_real+(chain_real[i]-avg_real)*(chain_real[i]-avg_real)
  s_real=s_real/N
  s_real=sqrt(s_real)
  uncertainty_real = s_real/sqrt(N)*(2*pi)**dim 
  return measurement_imag, uncertainty_imag, measurement_real, uncertainty_real  
#End
###################################

t=1
a=1
#U=6.0
#beta=5
mu=0
#nu_ext=(-3)*pi/beta
px_ext=pi/2.0
py_ext=pi/2.0
#kx1=ky1=0
#kx2=pi
#ky2=pi/3.0
step=20000000


for U in [1,2,3,4,5,6]:
  for beta in [5,4,3,2,1]:
    for n in range (0,10):
      nu_ext=(2*n+1)*pi/beta
      print
      print "Evaluation for the following parameters:"
      print 't = ', t
      print 'U = ', U
      print 'beta = ', beta
      print 'mu = ', mu
      print 'n = ', n
      print 'nu_ext = ', nu_ext
      print 'px_ext = ', px_ext
      print 'py_ext = ', py_ext
      print 'a = ', a
      print 'step = ', step
      print
      SIGMA_0=0
      SIGMA=0
      file_name='SCPT_with_feed_up_to_2nd_vs_iteration_n='+str(n)+'_U='+str(U)+'_beta='+str(beta)+'_pi_half_pi_half_MC=2e7.dat'
      file = open('result_SCPT_up_to_second_with_feedback_iter_2e7/'+file_name, 'w')  
      
      for i in range(0,50):
        print i+1
        m=1    # order of diagram
        DIM=m*2  # number of independent momentum variables: px and py
  
        first_order = naive_mc_Hartree(DIM,step)
        print 'Hartree = ', first_order[0], first_order[1]
        print '*******'
        print
        m=2    # order of diagram
        DIM=m*2  # number of independent momentum variables: px and py
        out_second_order = naive_mc_2nd(DIM,step)
        print 'Second order term: Imag part = ', out_second_order[0], out_second_order[1]
        print 'Second order term: Real part = ', out_second_order[2], out_second_order[3]
        print '*******'
        print
        if i==0:
          SIGMA_1 = U/2.0 + out_second_order[0]*1.0j 
        else:
          SIGMA_1 = first_order[0] + out_second_order[0]*1.0j + out_second_order[2]
        print 'Sigma_0 = ', SIGMA_0
        print 'Sigma_1 = ', SIGMA_1
        SIGMA=0.5*(SIGMA_0+SIGMA_1)
        print 'Sigma = ', SIGMA
        SIGMA_0=SIGMA
        txt = str(i+1) + '\t' + str(SIGMA.imag) + '\t' + str(SIGMA.real) + '\n'
        file.write(txt)
        print
      file.close()


