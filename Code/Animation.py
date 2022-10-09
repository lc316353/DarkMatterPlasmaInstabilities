# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:40:47 2022

@author: Schillings
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
import scipy.optimize as opt
plt.close("all")

##--Initialize the animation--##
dataName="PICResults"
buildAnimation=True
without=0 #0,1,2

header=np.loadtxt(dataName+".txt",max_rows=1)
rV=float(header[0])
N=int(header[1])
L=int(header[2])
tmax=int(header[3])
Nsteps=int(header[4])
J=int(header[5])
alpha=int(header[6])
rT=int(header[7])
deltat=tmax/Nsteps
data=np.loadtxt(dataName+".txt",skiprows=1) #np.zeros((2*Nsteps,N))
x=data[::2]
v=data[1::2]

times=np.linspace(0,Nsteps,(Nsteps+1))

Wkin,Wel,Wges=np.loadtxt(dataName+"Energy.txt")
E=np.loadtxt(dataName+"E.txt")

##--Initialize the snapshot-plot--##
Subplot=(True,(2,3),[0,2,6,10,14,20],((0,L),(-3*rV,3*rV)),"Scatterplots") #Scattermultiplot:(Plotbool, (rows, columns),[times],((xlim),(vlim)),savename)


##--Build the animation--##
if buildAnimation:
    ##--Organize the Subplots--##
    def setup_axes(fig, rect, rotation, axisScale, axisLimits):
        tr = Affine2D().scale(axisScale[0], axisScale[1]).rotate_deg(rotation)
    
        grid_helper = floating_axes.GridHelperCurveLinear(tr, extremes=axisLimits)
    
        ax = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
        fig.add_subplot(ax)
        aux_ax = ax.get_aux_axes(tr)
    
        return ax, aux_ax
    
    fig  = plt.figure(1, figsize=(10, 10))
    axes = []
    axisOrientation = [0, 0, 90, 0, 0]
    axisScale = [[6*rV,L],[1200*2.5,L],[2000*2.5,6*rV],[Wges[0]*1.1,tmax],[20,L]] #scales of the subplots
    axisPosition = [335,332,334,339,331]
    axisLimits = [(0, L, -3*rV, 3*rV),          #limits of the subplots
                  (0, L, 0, 1200),
                  (-3*rV,3*rV, 0, 2000),
                  (0,tmax,0,Wges[0]*1.1),
                  (0,L,-10,10)]
    
    label_axes = []
    for i in range(0, len(axisOrientation)-without):
        ax, aux_ax = setup_axes(fig, axisPosition[i], axisOrientation[i], 
                                axisScale[i], axisLimits[i])
        axes.append(aux_ax)
        label_axes.append(ax)
    
    label_axes[0].axis["bottom"].label.set_text('x') #setting labels
    label_axes[0].axis["right"].label.set_text('v')
    label_axes[0].axis["right"].toggle(ticklabels=True,label=True)
    label_axes[0].axis["left"].toggle(ticklabels=False,label=False)
    
    
    for i in range(1,len(label_axes)):
        for axisLoc in ['top','left','right']:
            label_axes[i].axis[axisLoc].set_visible(False) #remove axes
        label_axes[i].axis['bottom'].toggle(ticklabels=False)    
    
    fig.subplots_adjust(wspace=-0.30, hspace=-0.30, left=0.00, right=0.99, top=0.99, bottom=0.0)
    
    ##--Plot the animation--##
    def animate(i):
        i=int(i)
        for j in range(len(axes)):
            axes[j].clear()
        fig.suptitle(r"$r=${:.1f}, $N=${:.0e}, $t$={}$/\omega_p$".format(rV,N,i*tmax/Nsteps))
        axes[0].scatter(x[i,:],v[i,:],c="black",s=0.1,alpha=0.5)
        if without==0:
            axes[4].plot(np.linspace(0,L,J),E[i],color="purple")
        #maxpos=np.linspace(0,L,J)[np.argmax(np.abs(E[i][:int(J/2)])==max(np.abs(E[i][:int(J/2)])))]
        #axes[4].text(maxpos,8,"max {}".format(maxpos))
        #axes[0].set_xlim(0,L)
        #axes[0].set_ylim(-3*rV,3*rV)
        axes[1].hist(x[i],bins=int(N/400)+1,color="black")
        axes[1].set_xlim(0,L)
        axes[1].set_ylim(0,2*N/400)
        axes[2].hist(v[i],bins=int(N/400)+1,color="black")
        #axes[2].set_xlim(-3*rV,3*rV)
        #axes[2].set_ylim(0,3.5*N/400)
        if without<=1:
            axes[3].plot(times[0:i+1]*deltat,Wkin[0:i+1],color="blue")
            axes[3].plot(times[0:i+1]*deltat,Wel[0:i+1],color="orange")
            axes[3].plot(times[0:i+1]*deltat,Wges[0:i+1],color="green")
        
    ##--Initialize and Save--##
    animation=ani.FuncAnimation(fig, animate,times,interval=500*deltat)
    animation.save("PICanimation.gif")

##--Build a plot with several snapshots of the animation--##
if Subplot[0]:
    fig,ax=plt.subplots(*Subplot[1],figsize=(15,10))
    for i in range(Subplot[1][0]):
        for j in range(Subplot[1][1]):
            plt.subplot(Subplot[1][0],Subplot[1][1],i*Subplot[1][1]+j+1)
            plt.scatter(x[np.argmax(times*deltat>=Subplot[2][i*Subplot[1][1]+j])],v[np.argmax(times*deltat>=Subplot[2][i*Subplot[1][1]+j])],c="black",s=0.1,alpha=0.5)
            if j==0:
                plt.ylabel("v")
            if i==Subplot[1][0]-1:
                plt.xlabel("x")
            plt.xlim(*Subplot[3][0])
            plt.ylim(*Subplot[3][1])
            plt.title(r"$\omega_p t=$"+"{:.2f}".format(deltat*times[np.argmax(times*deltat>=Subplot[2][i*Subplot[1][1]+j])]))
    plt.savefig(Subplot[4]+".pdf")
            

"""
spaceFFTn=[]
ns=[]
for i in range(Nsteps):
    n,xn,waste=plt.hist(x[i],bins=int(N/400)+1,color="black")
    spaceFFTn.append(np.fft.fft(n))
    ns.append(n)
ns=np.array(ns)
spaceFFTn=np.array(spaceFFTn)
#FFTn=[]
#for i in range(N):
#    FFTn.append(np.fft.fft(spaceFFTn.T[i]))
growthRates=(spaceFFTn[1:,:]-spaceFFTn[:-1,:])/spaceFFTn[:-1,:]/deltat
print(np.max(growthRates))
"""
##--Fourier transform of the electric field--##
FFTE=[]
for i in range(Nsteps+1):
    FFTE.append(np.fft.fft(E[i]))
FFTE=np.array(FFTE)
k=np.linspace(0,len(E[1])-1,len(E[1]))*2*np.pi/L

##--Plot some wavemodes--##
kspec=[0.01,0.1,0.35,0.45,1]
omega=[]
plt.figure()
for ks in kspec:
    ind=np.argmax(k>ks)
    plt.plot(times[1:]*deltat,np.abs(FFTE[1:,ind]),label="k="+str(ks))
    omega.append(np.sum((np.abs(FFTE[5:,np.argmax(k>ks)])[1:]-np.mean(np.abs(FFTE[5:,np.argmax(k>ks)])))*(np.abs(FFTE[5:,np.argmax(k>ks)])[:-1]-np.mean(np.abs(FFTE[5:,np.argmax(k>ks)])))<0)/4/len(times[5:])/deltat*2*np.pi)

plt.title("wave modes")
plt.xlabel(r"$\omega_p t$")
plt.ylabel(r"$|E_k|$")
plt.legend()


def linear(x,a,b):
    return a*x+b

"""
kval=0.045
omegak=np.sum((np.abs(FFTE[5:,np.argmax(k>kval)])[1:]-np.mean(np.abs(FFTE[5:,np.argmax(k>kval)])))*(np.abs(FFTE[5:,np.argmax(k>kval)])[:-1]-np.mean(np.abs(FFTE[5:,np.argmax(k>kval)])))<0)/4/len(times[5:])/deltat*2*np.pi
maxima=[]
maximatimes=[]
plt.figure()
kdamp=0.045
plt.plot(times*deltat,np.abs(FFTE[:,np.argmax(k>kdamp)]))
start=8
for i in range(7):
    regionSize=int(2*np.pi/omega[np.argmax(kspec==kdamp)]/deltat/4)+2
    region=np.abs(FFTE[start:,np.argmax(k>kdamp)])[i*regionSize:(i+1)*regionSize]
    maxima.append(max(region))
    maximatimes.append(times[np.argmax(region>=max(region))+i*regionSize+start]*deltat)
    plt.axvline(times[start:][i*regionSize]*deltat,linestyle="--",color="black")

omegaR=0
DampingParams=opt.curve_fit(linear, maximatimes, np.log(maxima))
plt.plot(maximatimes,maxima,"ro")
plt.plot(times*deltat,np.exp(linear(times*deltat,*DampingParams[0])),"-.g",label=r"$\omega=${}+{:.3f}$i$".format(omegaR,DampingParams[0][0]))
plt.title("k={}".format(kdamp))
plt.xlabel(r"$t\omega_p$")
plt.ylabel(r"$E_k$")
DampingParams2=opt.curve_fit(linear, times[:55]*deltat, np.log(np.abs(FFTE[:55,np.argmax(k>kdamp)])))
plt.plot(times*deltat,np.exp(linear(times*deltat,*DampingParams2[0])),color="purple",linestyle="dashdot",label=r"$\omega=${}+{:.3f}$i$".format(omegaR,DampingParams2[0][0]))

plt.legend()
"""
#--Measure growth rate--##
kdamp=0.61/rV
start=10; end=200
omegaR=0
plt.figure()
plt.plot(times*deltat,np.abs(FFTE[:,np.argmax(k>kdamp)]))
plt.title("k={}".format(kdamp))
plt.xlabel(r"$t\omega_p$")
plt.ylabel(r"$E_k$")
plt.yscale("log")
DampingParams2=opt.curve_fit(linear, times[start:end]*deltat, np.log(np.abs(FFTE[start:end,np.argmax(k>kdamp)])))
plt.plot(times*deltat,np.exp(linear(times*deltat,*DampingParams2[0])),color="purple",linestyle="dashdot",label=r"$\omega=${}+{:.4f}$i$".format(omegaR,DampingParams2[0][0]))
plt.legend()


##--Frequency and damping fits--##

#k=np.loadtxt("k.txt")
#FFTE=np.loadtxt("Ek.txt").T

def abssin(t,omega,A,B,C):
    return np.abs(A*np.cos(omega*t+B))

def abssinexp(t,omegaR,omegaI,A,B,C):
    return np.abs(A*np.cos(omegaR*t+B))*np.exp(omegaI*t)+C

def oneoverk(k,A):
    return A/k
##--Single Fit--##
plt.figure()
kfit=0.45
exclusion1=10#np.argmax(times*deltat>2.5)
exclusion2=len(times)//4*3#np.argmax(times*deltat>10)
plt.title("k="+str(kfit))
ydata=np.abs(FFTE[:,np.argmax(k>kfit)])
xdata=times*deltat
plt.plot(xdata,ydata,"bo",label="data")
param=opt.curve_fit(abssin,xdata[exclusion1:exclusion2],ydata[exclusion1:exclusion2])
#p0=list(param[0]); p0.insert(1,0)
p0=[1.01,-0.01,1,0,0]
param2=opt.curve_fit(abssinexp,xdata[exclusion1:exclusion2],ydata[exclusion1:exclusion2],p0=p0)
#plt.plot(xdata,abssin(xdata,*param[0]),label=r"frequency only fit $\omega/\omega_p=${:.4f}".format(param[0][0]))
plt.plot(xdata,abssinexp(xdata,*param2[0]),label=r"full fit: $\omega/\omega_p=${:.4f}+{:.4f}i".format(param2[0][0],param2[0][1]),color="tab:orange")
plt.axvline(xdata[exclusion1],linestyle="--",color="black")
plt.axvline(xdata[exclusion2-1],linestyle="--",color="black")
plt.legend()

##--Automatized several fits--##
params=[]
fitfunc=abssinexp
start=1
end=57
p0=[1,0,1,0,0]
maximums=[]
firstmintime=[]
allfitQualitytest=True
firstminex=np.argmax(xdata>np.pi)
for i,kk in enumerate(k[start:end]):    
    params.append(opt.curve_fit(fitfunc,xdata[exclusion1:exclusion2],np.abs(FFTE[exclusion1:exclusion2,i+start]),p0=p0)[0])
    p0=params[i]
    maximums.append(max(np.abs(FFTE[exclusion1:exclusion2,i+start])))
    firstmintime.append(xdata[np.argmax(np.abs(FFTE)[1:firstminex,i+start]==min(np.abs(FFTE)[1:firstminex,i+start]))])
    
    if allfitQualitytest and (i in [1,35,68]):
        plt.figure()
        plt.title("k="+str(kk))
        plt.plot(xdata[1:],np.abs(FFTE[1:,np.argmax(k>=kk)]),"ob",label="data")
        plt.plot(xdata[1:],fitfunc(xdata,*params[i])[1:],label="fit")
        plt.xlabel(r"$\omega_p t$")
        plt.ylabel(r"$|E_k|$")
        plt.legend()
params=np.array(params)

##--Plot some parameters--##
plt.figure()
plt.title("Amplitude")
plt.plot(k[start:end],params[:,2])
#plt.plot(k[start:end],maximums)
#plt.plot(k[1:end],1/k[1:end])
oneoverkparam=opt.curve_fit(oneoverk,k[start+2:end],params[2:,2])
plt.plot(k[start:end],oneoverk(k[start:end],*oneoverkparam[0]),"--g",label="fit: {:.1f}/k".format(oneoverkparam[0][0]))
plt.xlabel(r"$k$")
plt.xscale("log")
plt.ylabel(r"$A$")
plt.yscale("log")
plt.legend()

plt.figure()
plt.title("Frequency")
plt.plot(k[start:end],params[:,0],label=r"$\omega_r$")
plt.plot(k[start:end],params[:,1],label=r"$\omega_i$")
#plt.plot(k[start:end],2*np.pi/4/np.array(firstmintime))
plt.xlabel(r"$k$")
plt.ylabel(r"$\omega$")
plt.legend()
plt.figure()
plt.title("k="+str(kfit))
plt.plot(xdata,np.abs(FFTE[:,np.argmax(k>kfit)]),"ob",label="data")
plt.plot(xdata,fitfunc(xdata,*params[np.argmax(k[start:end]>kfit)]),label="fit")
plt.legend()

#np.savetxt("k.txt",k)
#np.savetxt("Ek.txt",np.abs(FFTE.T))
    

"""
def abssin(t,omega,A,B,C,omega2,A2,B2):
    return np.abs(A*np.sin(omega*t+B)+A2*np.sin(omega2*t+B2))+C
def sin(t,omega,A,B,C):
    return A*np.sin(omega*t+B)+C
def sin2(t,omega,A,B,C):
    return A*np.sin(omega*t+B)**2+C

plt.figure()
plt.plot(times*deltat,Wel)
omegaE=np.sum((E[5:,10][1:]-np.mean(E[5:,10]))*(E[5:,10][:-1]-np.mean(E[5:,10]))<0)/2/len(times[5:])/deltat*2*np.pi
omegaW=np.sum((Wel[5:][1:]-np.mean(Wel[5:]))*(Wel[5:][:-1]-np.mean(Wel[5:]))<0)/4/len(times[5:])/deltat*2*np.pi

#param1=opt.curve_fit(sin,times[5:]*deltat,Wel[5:])
#param2=opt.curve_fit(sin2,times[5:]*deltat,Wel[5:])
#plt.plot(times*deltat,sin(times*deltat,*param1[0]))
#plt.plot(times*deltat,sin2(times*deltat,*param2[0]))
"""


"""
##--Analyse the velocity distribution by fitting the functions defined above--##
vhistx=(vhist[1][1:]+vhist[1][:-1])/2
param1=opt.curve_fit(MBV,vhistx,vhist[0])
param2=opt.curve_fit(TwoStream,vhistx,vhist[0])
plt.plot(vhistx,MBV(vhistx,*param1[0]),color="pink")
plt.plot(vhistx,TwoStream(vhistx,*param2[0]),color="orange")
"""

