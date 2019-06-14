'''
AMI+SCPT up to the second order

Last Updates: 04-June-2019
'''

###################################
##########Import libraries#########
###################################
#Begin
from random import random
from random import randint
import pickle
from cmath import *
import numpy as np
from Symbolic_multi_AMI_new import *
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
############Update momenta#########
###################################
#Begin
def Integrand(M,NUMBER):   # For now number=1
  out=[]  # To store output
  
  s_list=deepcopy(S_list[M-1][NUMBER-1])
  p_list_freq=deepcopy(P_list_freq[M-1][NUMBER-1])
  p_list_mnta=deepcopy(P_list_mnta[M-1][NUMBER-1])
  r_freq=deepcopy(R_list_freq[M-1][NUMBER-1])
  r_mnta=deepcopy(R_list_mnta[M-1][NUMBER-1])
  g_sym=deepcopy(G_sym_list[M-1][NUMBER-1])
  F=deepcopy(F_list[M-1][NUMBER-1])

  ###################################
  #####Initialize moemntum array#####
  ###################################
  #Begin
  k_init=[ [0]*2 for i in range (0,M+1) ]  # initialize momenta
  k_init[M][0] = px_ext
  k_init[M][1] = py_ext
  #print k_init
  #End
  ###################################
  
  ###################################
  ############AMI algebra############ 
  ###################################
  #Begin
  for i in range (0,M):  # Pick momenta randomly
    for j in range (0,2):
      k_init[i][j]= (2*pi)*random()-pi
  #k_init[0][0]=0
  #k_init[0][1]=0
  #k_init[1][0]=pi/4.0
  #k_init[1][1]=pi/3.0
  #k_init[2][0]=pi
  #k_init[2][1]=pi/3.0
  #print 'k_init = ', k_init
  #ener=-2*( cos(k_init[0][0]) + cos(k_init[0][1]) )
  #print fermi(ener)/4**M/pi**(2*M)
  poles = Poles(p_list_freq, p_list_mnta,g_sym[0],k_init,t,mu-SIGMA)
  for i in range (0,len(poles)):
    for j in range (0,len(poles[i])):
      if poles[i][j]==None:
        j = len(poles[i])+1
      else:
        for l in range (0,len(poles[i][j])):
          #print poles[i][j][k]
          poles[i][j][l] = f(poles[i][j][l],beta)  # Find the numerical values 
  #print 'poles = ', poles

  ########################################
  ################AMI Algebra#############
  ########################################

  res1 = dot_num(s_list[1][0], poles[1][0])
  temp=res1
  for it_m in range (2,M+1):
    o = 0
    for i in range(0, len(s_list[it_m])):
      if s_list[it_m][i] != None:
        o = o + 1
      else:
        i = len(poles[it_m]) + 1
    
    S_new = [None]*(o)
    P_new = [None]*(o)
    for i in range(0, len(s_list[it_m])):
      if s_list[it_m][i] != None:
        S_new[i] = s_list[it_m][i]
        P_new[i] = poles[it_m][i]
  
      else:
        i = len(s_list[it_m]) + 1

    res2 = dot_arr(S_new, P_new)
    temp = cross(res1,res2)
    res1 = temp
  
  G_val = 0.0  # To store numerical value at a given external frequency
  if len(r_freq[0])!=0:
    for j in range (0, len(temp)):
      G_val = temp[j]*G_eval(r_freq[j], r_mnta[j], nu_ext, g_sym[0],k_init,t,mu-SIGMA,beta) + G_val
    #print 'G_val = ', G_val
    #print '?????'
  else:
    for j in range (0, len(temp)):
      G_val = temp[j] + G_val 
    #print '?????'   
    #print 'G_val = ', G_val
  #print -G_val/4/4/4/pi/pi/pi/pi/pi/pi
  out.append((-1)**(M+F)*(U**M)*G_val.real/4**M/pi**(2*M))
  out.append((-1)**(M+F)*(U**M)*G_val.imag/4**M/pi**(2*M))
 
  #print out
  return out
      
#End
###################################

###################################
########Naive Monte Carlo##########
###################################
#Begin
def naive_mc(M,NUMBER):
  avg_re=0
  avg_im=0
  ch_re=[]
  ch_im=[]
  for i_T in range(0,T):
    for i_N in range(0,N):
      result=Integrand(M,NUMBER)
      re=result[0]
      im=result[1]
      avg_re=avg_re+re
      avg_im=avg_im+im
    ch_re.append(avg_re/N)
    ch_im.append(avg_im/N)
    #print 're = ', avg_re/N
    #print 'im = ', avg_im/N
    avg_re=0 
    avg_im=0

  re=0
  im=0
  for i_T in range(0,T):
    re=re+ch_re[i_T]
    im=im+ch_im[i_T]
  re_fin = (2*pi)**(2*M)*re/T
  im_fin = (2*pi)**(2*M)*im/T
  var_re=0
  var_im=0
  for i_T in range(0,T):
    var_re=var_re+(re_fin-(2*pi)**(2*M)*ch_re[i_T])**2
    var_im=var_im+(im_fin-(2*pi)**(2*M)*ch_im[i_T])**2
  var_re = sqrt(var_re/(T-1.0))
  var_im = sqrt(var_im/(T-1.0))
  er_re = var_re/sqrt(T)
  er_im = var_im/sqrt(T) 
  #print 'Real Part = ', re_fin, '+-', er_re
  #print 'Imaginary Part = ', im_fin, '+-', er_im 
  #txt=str(M) + '\t' + str(i_files) + '\t' + str(N) + '\t' + str(T) + '\t' + str(re_fin) + '\t' + str(er_re) + '\t' + str(im_fin) + '\t' + str(er_im) + '\n'
  #file.write(txt)
  #file.close()
  return im_fin, er_im, re_fin, er_re
#End
###################################

###################################
##########Read AMI arrays##########
###################################
#Begin
S_list=[[None]*2 for i in range(0,3)]
P_list_freq=[[None]*2 for i in range(0,3)]
P_list_mnta=[[None]*2 for i in range(0,3)]
R_list_freq=[[None]*2 for i in range(0,3)]
R_list_mnta=[[None]*2 for i in range(0,3)]
G_sym_list=[[None]*2 for i in range(0,3)]
F_list=[[None]*2 for i in range(0,3)]

number=1
for m in range (1,3):
  print 'm = ', m
  file_name_S = 'S_m_'+str(m)+'_num_'+str(number)
  with open(file_name_S,"rb") as fp:
    s_arr=pickle.load(fp)
  S_list[m-1][number-1]=deepcopy(s_arr)

  file_name_P_freq = 'P_freq_m_'+str(m)+'_num_'+str(number)
  with open(file_name_P_freq,"rb") as fp:
    p_arr_freq=pickle.load(fp)
 
  P_list_freq[m-1][number-1]=deepcopy(p_arr_freq)

  file_name_P_mnta = 'P_mnta_m_'+str(m)+'_num_'+str(number)
  with open(file_name_P_mnta,"rb") as fp:
    p_arr_mnta=pickle.load(fp)
  
  P_list_mnta[m-1][number-1]=deepcopy(p_arr_mnta)

  file_name_R_freq = 'R_freq_m_'+str(m)+'_num_'+str(number)
  with open(file_name_R_freq,"rb") as fp:
    r_arr_freq=pickle.load(fp)
  
  R_list_freq[m-1][number-1]=deepcopy(r_arr_freq)

  file_name_R_mnta = 'R_mnta_m_'+str(m)+'_num_'+str(number)
  with open(file_name_R_mnta,"rb") as fp:
    r_arr_mnta=pickle.load(fp)
  
  R_list_mnta[m-1][number-1]=deepcopy(r_arr_mnta)


  file_name_G_sym = 'G_sym_m_'+str(m)+'_num_'+str(number)
  with open(file_name_G_sym,"rb") as fp:
    g_arr_sym=pickle.load(fp)
  G_sym_list[m-1][number-1]=deepcopy(g_arr_sym)
  
  
  ###################################
  #######Read number of loops########
  ###################################
  #Begin
  file_name_F='f_m_'+str(m)+'_num_'+str(number)
  with open(file_name_F,"rb") as fp:
    F_arr=pickle.load(fp)  # F: number of loops
  F_list[m-1][number-1]=F_arr
  #End
  ###################################
m=3
for number in range (1,3):
  print 'm = ', m
  file_name_S = 'S_m_'+str(m)+'_num_'+str(number)
  with open(file_name_S,"rb") as fp:
    s_arr=pickle.load(fp)
  S_list[m-1][number-1]=deepcopy(s_arr)

  file_name_P_freq = 'P_freq_m_'+str(m)+'_num_'+str(number)
  with open(file_name_P_freq,"rb") as fp:
    p_arr_freq=pickle.load(fp)
 
  P_list_freq[m-1][number-1]=deepcopy(p_arr_freq)

  file_name_P_mnta = 'P_mnta_m_'+str(m)+'_num_'+str(number)
  with open(file_name_P_mnta,"rb") as fp:
    p_arr_mnta=pickle.load(fp)
  
  P_list_mnta[m-1][number-1]=deepcopy(p_arr_mnta)

  file_name_R_freq = 'R_freq_m_'+str(m)+'_num_'+str(number)
  with open(file_name_R_freq,"rb") as fp:
    r_arr_freq=pickle.load(fp)
  
  R_list_freq[m-1][number-1]=deepcopy(r_arr_freq)

  file_name_R_mnta = 'R_mnta_m_'+str(m)+'_num_'+str(number)
  with open(file_name_R_mnta,"rb") as fp:
    r_arr_mnta=pickle.load(fp)
  
  R_list_mnta[m-1][number-1]=deepcopy(r_arr_mnta)


  file_name_G_sym = 'G_sym_m_'+str(m)+'_num_'+str(number)
  with open(file_name_G_sym,"rb") as fp:
    g_arr_sym=pickle.load(fp)
  G_sym_list[m-1][number-1]=deepcopy(g_arr_sym)
  
  
  ###################################
  #######Read number of loops########
  ###################################
  #Begin
  file_name_F='f_m_'+str(m)+'_num_'+str(number)
  with open(file_name_F,"rb") as fp:
    F_arr=pickle.load(fp)  # F: number of loops
  F_list[m-1][number-1]=F_arr
  #End
  ###################################
data = np.loadtxt('ext_vars.dat')  # reading external variables from a file
count = len(open('ext_vars.dat').readlines())
if count==1:
  EXT_VARS = [None]*count
  EXT_VARS[0]=data
if count>1:
  EXT_VARS = [None]*count  
  for i in range (0, count):
    EXT_VARS[i] = data[i,:]  
#print EXT_VARS
t=EXT_VARS[1][0]
U=EXT_VARS[1][1]
beta=EXT_VARS[1][2]
mu=EXT_VARS[1][3]
nu_ext=(2*EXT_VARS[1][4]+1)*pi/beta
px_ext=EXT_VARS[1][5]
py_ext=EXT_VARS[1][6]
lat_const=EXT_VARS[1][7]
mu=1.0
#print
#print "Evaluation for the following parameters:"
#print 't = ', t
#print 'U = ', U
#print 'beta = ', beta
#print 'mu = ', mu
#print 'nu_ext = ', nu_ext
#print 'px_ext = ', px_ext
#print 'py_ext = ', py_ext
#print 'a = ', lat_const
T=50
N=20000
#print 'T = ', T
#print 'N = ', N
#print naive_mc(2,1)
for U in [1]:
  for beta in [5]:
    for n in range (1,2):
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
      print 'a = ', lat_const
      print 'T = ', T
      print 'N = ', N
      print
      er_SIGMA_0=0
      er_SIGMA=0
      SIGMA_0=0
      SIGMA=0
      file_name='AMI+SCPT_mu=1_with_feed_up_to_3rd_vs_iteration_n='+str(n)+'_U='+str(U)+'_beta='+str(beta)+'_pi_half_pi_half_MC=1e6.dat'
      file = open('result_AMI+SCPT_up_to_third_with_feedback_iter_1e6_mu=1/'+file_name, 'w')  
      
      for i in range(0,50):
        print i+1
        m=1    # order of diagram
        DIM=m*2  # number of independent momentum variables: px and py
  
        first_order = naive_mc(1,1)
        print 'Hartree = ', 1j*first_order[0] + first_order[2]
        print '*******'
        print
        m=2    # order of diagram
        DIM=m*2  # number of independent momentum variables: px and py
        out_second_order = naive_mc(2,1)
        print 'Second order term: Imag part = ', out_second_order[0], out_second_order[1]
        print 'Second order term: Real part = ', out_second_order[2], out_second_order[3]
        print '*******'
        print
        out_third_order_1=naive_mc(3,1)
        print 'Third order term_1: Imag part = ', out_third_order_1[0], out_third_order_1[1]
        print 'Third order term_1: Real part = ', out_third_order_1[2], out_third_order_1[3]
        print '*******'
        print
        out_third_order_2=naive_mc(3,2)
        print 'Third order term_2: Imag part = ', out_third_order_2[0], out_third_order_2[1]
        print 'Third order term_2: Real part = ', out_third_order_2[2], out_third_order_2[3]
        print '*******'
        print
        if i==0:
          SIGMA_1 = first_order[2] + out_second_order[0]*1.0j + out_second_order[2] + out_third_order_1[0]*1.j + out_third_order_1[2] + out_third_order_2[0]*1.0j + out_third_order_2[2]
          #SIGMA_1 = U/2.0 + out_second_order[0]*1.0j #+ out_third_order_1[0]*1.j + out_third_order_1[2] + out_third_order_2[0]*1.0j + out_third_order_2[2]
          #er_SIGMA_1 = out_second_order[1]*1.0j #+ out_third_order_1[1]*1.0j + out_third_order_1[3] + out_third_order_2[1]*1.0j + out_third_order_2[3]
          er_SIGMA_1 = 1j*first_order[1] + first_order[3] + out_second_order[1]*1.0j + out_second_order[3] + out_third_order_1[1]*1.0j + out_third_order_1[3] + out_third_order_2[1]*1.0j + out_third_order_2[3]
        else:
          SIGMA_1 = first_order[0]*1.0j + first_order[2] + out_second_order[0]*1.0j + out_second_order[2] + out_third_order_1[0]*1.j + out_third_order_1[2] + out_third_order_2[0]*1.0j + out_third_order_2[2]
          er_SIGMA_1 = 1j*first_order[1] + first_order[3] + out_second_order[1]*1.0j + out_second_order[3] + out_third_order_1[1]*1.0j + out_third_order_1[3] + out_third_order_2[1]*1.0j + out_third_order_2[3]
        print 'SIGMA_0 = ', SIGMA_0
        print 'er_SIGMA_0 = ', er_SIGMA_0 
        print 'SIGMA_1 = ', SIGMA_1
        print 'er_SIGMA_1 = ', er_SIGMA_1 
        SIGMA=0.5*(SIGMA_0+SIGMA_1)
        er_SIGMA = 0.5*(er_SIGMA_0+er_SIGMA_1)
        print 'SIGMA = ', SIGMA
        print 'er_SIGMA = ', er_SIGMA
        SIGMA_0=SIGMA
        er_SIGMA_0=er_SIGMA
        txt = str(i+1) + '\t' + str(SIGMA.imag) + '\t' + str(er_SIGMA.imag) + '\t' + str(SIGMA.real) + '\t' + str(er_SIGMA.real) + '\n'
        file.write(txt)
        print
      file.close()
#print S_list
#print F_list
#print Integrand(1,1) 
