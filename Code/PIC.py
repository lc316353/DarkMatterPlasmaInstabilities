# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 19:55:40 2022

@author: Schillings
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sc
import scipy.optimize as opt
from functools import partial
from joblib import Parallel,delayed
from time import time as realtime
plt.close("all")


##~~~~~Relic Functions not used, but I am worried to delete them~~~~##
##--This function generates random velocities from two thermal clusters that are moving through each other--##
def drawTwoStream(N,rV,alpha=1,rT=1):
    seed=np.random.uniform(0,1,N)
    allv=np.linspace(-10*rV,10*rV,10000)
    ret=[]
    for ss in seed:
        ret.append(np.argmax(((1-alpha/2)*(1-sc.erf((rV-allv)/np.sqrt(2)))+alpha/2*(1+sc.erf((rV+allv)*rT/np.sqrt(2))))/2>ss))
    return allv[ret]

def drawTwoStreamMP(N,rV,alpha=1,rT=1):
    seed=np.random.uniform(0,1,N)
    allv=np.linspace(-10*rV,10*rV,10000)
    ret=Parallel(n_jobs=-1)(delayed(draw2)(ss,rV,alpha,rT,allv) for ss in seed)
    return allv[ret]

def draw2(seed,rV,alpha,rT,allv):
    return np.argmax(((1-alpha/2)*(1-sc.erf((rV-allv)/np.sqrt(2)))+alpha/2*(1+sc.erf((rV+allv)*rT/np.sqrt(2))))/2>seed)

##--This function generates random velocities from one thermal cluster--##
def drawMBV(N):
    seed=np.random.uniform(0,1,N)
    allv=np.linspace(-10,10,10000)
    ret=[]
    for ss in seed:
        ret.append(np.argmax((1+sc.erf(allv/np.sqrt(2)))/2>ss))
    return allv[ret]

##--This draws N positions from a uniform distribution with a wave perturbance with wavelength 2pi/k and amplitude a--##   
def drawWave(N,k,a):
    seed=np.random.uniform(0,1,N)
    allx=np.linspace(0,L,10000)
    ret=[]
    for ss in seed:
        ret.append(np.argmax((allx+a/k*np.sin(k*allx))/(L+a/k*np.sin(k*L))>ss))
    return allx[ret]

##--This is the Maxwell-Boltzmann distribution with V=sqrt(T/m)--##
def MBV(v,V,A):
    return A*np.exp(-v**2/2/V**2)

##--This defines two Maxwell-Boltzmann distributions streaming through each other with v0--##
def TwoStream(v,v0,V,A):
    return A/2*(np.exp(-(v-v0)**2/2/V**2)+np.exp(-(v+v0)**2/2/V**2))


##~~~~Functions used~~~~##

##--This draws positions from a uniform distribution with a wave perturbance with wavelength 2pi/k and amplitude a--##
def drawWaveMP(N,k,a):
    seed=np.random.uniform(0,1,N)
    allx=np.linspace(0,L,10000)
    ret=Parallel(n_jobs=-1)(delayed(drawX)(ss,k,a,allx) for ss in seed)
    return allx[ret]

def drawX(seed,k,a,allx):
    return np.argmax((allx+a/k*np.sin(k*allx))/(L+a/k*np.sin(k*L))>seed)

##--This draws N velocities from a Maxwell-Boltzmann distribution with velocity shift rV and dispersion rT^2--##
def drawMBDistribution(N,rV=0,rT=1):
    seed=np.random.uniform(0,1,N)
    allv=np.linspace(-10*np.abs(rV)-10,10*np.abs(rV)+10,10000)
    ret=Parallel(n_jobs=-1)(delayed(drawV)(ss,rV,rT,allv) for ss in seed)
    return allv[ret]

def drawV(seed,rV,rT,allv):
    return np.argmax((1-sc.erf((rV-allv)*rT/np.sqrt(2)))/2>seed)

##--Ion background for more adventureous setups--##
def TwoBlops(J,mu1,mu2,sigma,rV,t):
    xd=np.linspace(0,J-1,J)*deltax
    periods=10
    blop1=np.zeros((len(xd)))
    blop2=np.zeros((len(xd)))
    for i in range(-periods,periods):
        blop1+=np.exp(-(xd-i*L-mu1-rV*t)**2/2/sigma**2)
        blop2+=np.exp(-(xd-mu2+rV*t-i*L)**2/2/sigma**2)
    return (1-alpha/2)/np.sqrt(2*np.pi*sigma**2)*blop1+alpha/2/np.sqrt(2*np.pi*sigma**2)*blop2

def TwoBoxes(J,mu1,mu2,sigma,rV,t):
    xd=np.linspace(0,J-1,J)*deltax
    return (1-alpha/2)/sigma*(abs(xd-mu1-rV*t)<sigma/2)+alpha/2/sigma*(abs(xd-mu2+rV*t)<sigma/2)

##--Functions that return the right setup depending on input key word and parameters--##
def drawPositions(distribution,N,L,mu1,mu2,sigma,k,a):
    distribution=distribution.lower().replace(" ","")
    if distribution=="uniform" or distribution=="":
        return np.random.uniform(0,L,N)
    if distribution=="wave":
        return drawWaveMP(N, k, a)
    if distribution=="twoblops":
        x1=np.random.normal(mu1,sigma,int(N*alpha/2))
        x2=np.random.normal(mu2,sigma,int(N*(1-alpha/2)))
    elif distribution=="twoboxes":
        x1=np.random.uniform(mu1-sigma/2,mu1+sigma/2,int(N*alpha/2))
        x2=np.random.uniform(mu2-sigma/2,mu2+sigma/2,int(N*(1-alpha/2)))
    return np.array(list(x2)+list(x1))
        
def drawVelocities(distribution,N,rV,alpha,rT):
    distribution=distribution.lower().replace(" ","")
    if distribution=="uniform":
        return np.zeros((N))
    elif distribution=="mbv" or distribution=="onestream" or distribution=="maxwell-boltzmann" or distribution=="maxwellboltzmann":
        return drawMBDistribution(N,0)#return drawMBV(N)
    elif distribution=="twostream":
        v1=drawMBDistribution(int(alpha/2*N),rV)
        v2=drawMBDistribution(N-int(alpha/2*N),-rV,rT)
        return np.array(list(v2)+list(v1))#drawTwoStreamMP(N, rV, alpha, rT)
    
def BackgroundIons(distribution,parameters):
    distribution=distribution.lower().replace(" ","")
    if distribution=="uniform" or distribution=="" or distribution=="wave":
        return 1
    elif distribution=="twoblops":
        return TwoBlops(*parameters)*N/n0
    elif distribution=="twoboxes":
        return TwoBoxes(*parameters)*N/n0
    
    
def lin(x,a,b):
    return a*x+b
    

##--The parameters of the PIC simulation--##
rV=10                                               #v0/V1 - streaming velocity normalized to the start temperature-mass ratio
alpha=1                                             #n1/n2 - ratio of number densities
rT=1                                                #sqrt(T1/T2)=V1/V2 - ratio of stream velocity dispersions
N=int(2e4)                                          #The number of particles involved

L=100                                               #The length of the space before it loops (needs to be even)
J=100                                               #The number of grid points in space
deltax=L/J                                          #Distance between two space-gridpoints
n0=N/L                                              #Mean number density=background ion number density

jspace=np.linspace(0,J-1,J)
Fourier=np.exp(-1j*2*np.pi/J*jspace)                #Fourier transformation factor

tmax=35                                             #End time of the simulation in 1/omega_p 
Nsteps=int(10*tmax)                                 #Number of calculation steps in time
deltat=tmax/Nsteps                                  #time between calculation steps

distribution="uniform"                              #Initial distribution of PIC-particles ["uniform","wave","two blops","two boxes"]
background=distribution                             #Background distribution. Should be same as distribution.
vDistribution="two stream"                          #Initial velocity distribution ["uniform","one stream", "two stream"]

##--Parametes for wave initialization
kin=0.045#/rV                                       #The wavenumber of the wave initialized
a=0.2                                               #The relative amplitude of the wave

##--Parameters for two-blop- and two-boxes-initialization--##
mu1=L/2                                             #The positions of the two blops
mu2=3*L/4
sigma=L/25                                          #The width of the two blops

##--Parameters for reset initialization--##
reset=False                                         #Determines if one of the streams will be resetted repeatedly
BulletSize=L*100 #~1 Mpc
tcross=13 #sigma/rV

Nparts=BulletSize/L
partcrossingtime=1#L/rV#tcross/Nparts               #The time after which the stream will be resetted

##--Statisics--##
NoR=1                                               #Number of Runs of the simulation with new random seed
meanWtransfer=[]                                    
maxWtransfer=[]
ampli=[]

##--Output--##
write=True                                          #Determines if the results will be written into a file
NoDP=0                                              #"Number of Drone Points" - The path of NoDP points will be recorded through time
useSnapshot=False                                   #Use a former snapshot of x and v instead of random initialization
snapshotTime=2*tmax                                 #The time where a snapshot of x and v is saved
saveName="PICResults"                               #If write, then x,v(t) will be saved in saveName+".txt", the electric field in saveName+"E.txt" and the energy in saveName+"Energy.txt"
maximumSaveName="saturationTimes"                   #If NoR>1: saves the time of the fist maximum in the electric energy and several other things (see line 452)

plotProgress=True                                   #Plot the phase space (or other quantities) at some timesteps
plottransferevolution=False
plottemperatureevolution=True
plotgrowthrate=True


###--Start of the program--###
if(write):
    rTFile=open(saveName+"rT.txt","w")
    rTFile.write("{} {} {} {} {}\n".format("rT","V1growth","V2growth","V1err","V2err"))

starttime=realtime()
if True: #for N in [2000,20000,200000]:#for rT in np.arange(rT,rTmax+rTstep*0.1,rTstep): #This line can be used to loop through a parameter
    print("a="+str(a))
    
    V1sum=np.zeros((Nsteps+1)); V2sum=np.zeros((Nsteps+1))    #Temperature statistics
    V1all=[];V2all=[]
    allgrowths1=[]
    allgrowths2=[]
    allWkin=[]; allWel=[]
    sumWkin=np.zeros((Nsteps+1)); sumWel=np.zeros((Nsteps+1))
    k=np.linspace(0,J-1,J)*2*np.pi/L
    
    allgrowthrates=[]
    growthratesyserr=[]
    kinAmplitude=[]
    kinAmplituden=[]
    nmean=[]
    ##--Initialize what will be calculated--##
    for loops in range(NoR): 
        #rT=1+4*loops/NoR
        snapshot=np.loadtxt("snapshot.txt")
        
        st=realtime()
        if useSnapshot:
            x=snapshot[0]
            v=snapshot[1]
        else:
            x=drawPositions(distribution,N,L,mu1,mu2,sigma,kin,a)         #The position of every particle
            v=drawVelocities(vDistribution, N, rV, alpha, rT)           #The velocity of every particle
            #np.random.shuffle(v)
            #if vDistribution.lower().replace(" ","")=="twostream":
            #    v.sort()
        
        print("time1:",realtime()-st)
        x0=x.copy()                                                 #For later reset
        v0=v.copy()
        
        E=np.zeros((J))                                     #The electrical field at every grid point
        Phi=np.zeros((J+2))                                 #The electric potential at every grid point
        
        n=np.zeros((J))                                     #The number density at every grid point
        rho=np.zeros((J))                                   #n(x)/n0-1 number density fluctuation at every grid point
        rhohat=np.zeros((J),dtype=np.complex128)            #rho in Fourier space
        phihat=np.zeros((J),dtype=np.complex128)            #Phi in Fourier space
        
        
        ##--Some additional analysis--##
        Wkin=[np.sum(1/2*v**2)]                             #Total kinetic energy normalized to the mass
        Wel=[0]                                             #Total electric energy normalized to n0 and the starting temperature
        V1=[1]
        V2=[1/rT]
        N1=N-int(alpha/2*N)
        FFTE=[]
        FFTn=[]
                                                      
        colors=["black"]*N; sizes=[1]*N
        if NoDP>0:
            colors[-NoDP:]=["blue","red","yellow","green","orange"][:NoDP]
            sizes[-NoDP:]=[50]*NoDP
            xN1=[x[-NoDP:]]                                     
            vN1=[v[-NoDP:]]
        
        if(write):
            dataFile=open(saveName+".txt","w")
            Efile=open(saveName+"E.txt","w")
            dataFile.write("{} {} {} {} {} {} {} {}\n".format(rV,N,L,tmax,Nsteps,J,alpha,rT))
            np.savetxt(dataFile,[x,v],fmt='%.4e')
            
            
##--Start of the simulation--##
        for timesteps in range(Nsteps+1):
            #~Calculate the number density at each grid point by counting particles between two points and splitting proportional to position between the points~#
            n=np.zeros((J))
            jj=np.floor(x/deltax).astype(int)%J
            if(J>N):
                for i in range(N):
                    j=int(np.floor(x[i]/deltax))%J
                    n[j]+=((j+1)*deltax-x[i])/deltax**2
                    n[(j+1)%J]+=(x[i]-j*deltax)/deltax**2
            else:
                partx=((jj+1)*deltax-x)/deltax
                for j in range(J):
                    n[j]+=np.sum(partx[j==jj]/deltax)
                    n[(j+1)%J]+=np.sum((1-partx[j==jj])/deltax)
                
            #~Transforming the density fluctuations to Fourier space and calculating the potential with the Poisson equation~#
            rho=n/n0-BackgroundIons(background, [J,mu1,mu2,sigma,rV,timesteps*deltat])
            for j in range(1,J):
                rhohat[j]=np.sum(rho*Fourier**j)/J
            phihat[1:J//2+1]=-rhohat[1:J//2+1]/np.linspace(0,J-1,J)[1:J//2+1]**2/(2*np.pi)**2*L**2
            phihat[int(J/2)+1:]=np.conjugate(phihat[int(J/2)-1:0:-1])
            for j in range(J):
                Phi[j+1]=np.sum(phihat*Fourier**(-j))
                
            #~Calculate the electric field from the potential~#
            Phi[0]=Phi[-2]
            Phi[-1]=Phi[1]
            E=(Phi[:-2]-Phi[2:])/2/deltax
            
            #~Find the E-field at the position of each particle by linear interpolation~#
            Eatx=(x/deltax-jj)*E[(jj+1)%J]+((jj+1)-x/deltax)*E[jj]
                
            #~Calculate new velocities and postions of every particle with the equations of motion~#
            v=v-Eatx*deltat
            x=(x+v*deltat)%L
            
            ##--Append what is traced through time--##
            if NoDP>0:
                xN1.append(x[-NoDP:])
                vN1.append(v[-NoDP:])
            Wkin.append(np.sum(1/2*v**2))
            Wel.append(np.sum(1/2*E**2*deltax*n0))
            V1.append(np.std(v[:N1]))#v[:np.argmax(v>0)]))
            V2.append(np.std(v[N1:]))#v[:np.argmax(v>0)-1:-1]))
            FFTE.append(np.fft.fft(E))
            FFTn.append(np.fft.fft(n))
            
            if timesteps*deltat==snapshotTime:
                np.savetxt("snapshot.txt",[x,v])
                
            if write:
                np.savetxt(dataFile,[x,v],fmt='%.4e')
                np.savetxt(Efile,[E],fmt='%.4e')
            
            ##--Reset one of the streams after every partcrossingtime--##
            if((timesteps+1)%int(partcrossingtime/deltat)==0) and reset:
                x[:int(N/2)]=x0[:int(N/2)]
                v[:int(N/2)]=v0[:int(N/2)]
                
            
            ##--Plot the phase space and other simulation properties (E-field, number density, velocity distribution) and certain points in time--##            
            if (timesteps in [int(Nsteps*fa) for fa in [1,3/4,1/2,3/8,5/16,1/4,3/16,1/16,0]]) and (NoR==1) and plotProgress:
                print(timesteps)
                
                plt.figure()
                plt.scatter(x,v,s=sizes,c=colors,alpha=0.5)
                plt.xlabel(r"$x$")
                plt.ylabel(r"$v$")
                plt.title(r"$t=${:.1f}/$\omega_p$".format(timesteps*deltat))
                
                
                #plt.figure()
                #plt.title("E-field, $t=${:.1f}/$\omega_p$".format(timesteps*deltat))
                #plt.plot(np.linspace(0,J-1,J)*deltax,E,color="purple")
                
                #plt.figure()
                #plt.title("number density, $t=${:.1f}/$\omega_p$".format(timesteps*deltat))
                #plt.plot(np.linspace(0,J-1,J)*deltax,n/n0)
                #plt.plot(np.linspace(0,J-1,J)*deltax,BackgroundIons(background, [J,mu1,mu2,sigma,rV,timesteps*deltat]))
                
                #plt.figure()
                #vhist=plt.hist(v,bins=int(N/400)+1)
                #plt.title(r"distribution of $v$, $t=${:.1f}/$\omega_p$".format(timesteps*deltat))
                #plt.xlabel(r"$v$")
                
                #print("x: ",x,"\nv: ",v,"\nE: ",E,"\nPhi: ",Phi,"\nphihat: ",phihat,"\nn: ",n)
##--End of the simulation--##
            
        ##--Convenient energy arrays--##
        Wkin=(np.array(Wkin[:-1])+np.array(Wkin[1:]))/2; Wel=np.array(Wel[1:])
        Wges=Wkin+Wel
        time=np.linspace(0,Nsteps,Nsteps+1)*deltat
        sumWkin+=Wkin; sumWel+=Wel
        allWkin.append(Wkin); allWel.append(Wel)
        
        FFTE=np.array(FFTE)
        FFTn=np.array(FFTn)
        kinAmplitude.append(np.abs(FFTE[0,np.argmax(k>=kin)]))
        kinAmplituden.append(np.abs(FFTn[0,np.argmax(k>=kin)]))
        nmean.append(FFTn[0,np.argmax(k>=kin)])
        
        
        ##--Energy transfer: Calculate the mean over time--##
        transferstart=np.argmax(Wkin<1e6)
        meanWtransfer.append(1-np.sum(Wkin[transferstart:])/len(Wkin[transferstart:])/Wges[0])
        maxWtransfer.append(1-(max(Wel)-Wel[0])/Wges[0])
        ampli.append((max(Wel[:int(6/deltat)])-min(Wel[:int(6/deltat)]))/(max(Wel[:int(6/deltat)]+min(Wel[:int(6/deltat)]))))
        if write:
            dataFile.close()
            Efile.close()
            np.savetxt(saveName+"Energy.txt",[Wkin,Wel,Wges])
            np.savetxt(saveName+"rT={:.1f}.txt".format(rT),[V1,V2])
            
        
        ##--Calculate and plot the maximum growth rate--##
        kdamp=0.61/rV
        start=min(40,Nsteps-2); end=min(120,Nsteps) #TODO: work out
        DampingParams2=opt.curve_fit(lin, time[start:end], np.log(np.abs(FFTE[start:end,np.argmax(k>=kdamp)])))
        allgrowthrates.append(DampingParams2[0][0])
        growthratesyserr.append(DampingParams2[1][0,0])
        if plotgrowthrate:
            omegaR=0
            plt.figure()
            plt.plot(time,np.abs(FFTE[:,np.argmax(k>kdamp)]))
            plt.title("k={}".format(kdamp))
            plt.xlabel(r"$t\omega_p$")
            plt.ylabel(r"|$E_k$|")
            plt.yscale("log")
            plt.plot(time,np.exp(lin(time,*DampingParams2[0])),color="purple",linestyle="dashdot",label=r"$\omega=${}+{:.3f}$i$".format(omegaR,DampingParams2[0][0]))
            plt.legend()
        
        
        ##--Plot the temperature evolution during the simulation--##
        
        #allgrowths1.append((V1[int(partcrossingtime/deltat)]-V1[0])/partcrossingtime)
        #allgrowths2.append((V2[int(partcrossingtime/deltat)]-V2[0])/partcrossingtime)
        V1,V2=np.array(V1)[1:],np.array(V2)[1:]
        """
        plt.figure()
        plt.title(r"Temperature evolution, $r_{T,0}$="+"{:.1f}".format(rT))
        plt.xlabel(r"$t\omega_p$")
        plt.ylabel(r"$V/V_{1,0}$")
        plt.plot(time,V1,label=r"$V_{1}$")
        plt.plot(time,V2,label=r"$V_{2}$")
        plt.plot(time,V2/V1,label="temperature ratio")
        if np.argmax(Wkin/Wkin[0]<0.98)*deltat>1:
            plt.axvline(np.argmax(Wkin/Wkin[0]<0.98)*deltat,color="black",linestyle="dashed",label="up to here quite okay")
        plt.legend()
        """
        V1sum+=V1
        V2sum+=V2
        V1all.append(V1)
        V2all.append(V2)
        
    
        print("time2:",realtime()-st)
        print(loops)
##End of statistics--##
        
    ##--Plot the energies vs. time--##
    plt.figure()
    plt.title("Total energy")
    plt.xlabel(r"$t\omega_p$")
    plt.ylabel(r"$W/T$")
    if NoR==1:
        plt.plot(time,sumWkin/NoR,label="Total kinetic energy")
        plt.plot(time,sumWel/NoR,label="Total electric energy")
    else:
        Wkinerr=[];Welerr=[]; allWkin=np.array(allWkin); allWel=np.array(allWel)
        for i in range(Nsteps+1):
            Wkinerr.append(np.std(allWkin[:,i]))
            Welerr.append(np.std(allWel[:,i]))
        plt.errorbar(time,sumWkin/NoR,Wkinerr)
        plt.errorbar(time,sumWel/NoR,Welerr)
        noiseFile=open(maximumSaveName+".txt","a")
        Wmax=np.argmax(sumWel==max(sumWel))
        np.savetxt(noiseFile,["{:.2e} {:.4e} {:.0f} {:.2e} {:.4f} {:.4e} {:.4e} {:.4e} {:.4e} {:.4e} {:.4e} {:.4e} {:.4e} {:.4e} {:.4e}".format(N,time[Wmax],NoR,a,kin,np.mean(kinAmplitude),np.std(kinAmplitude)/np.sqrt(NoR),np.mean(kinAmplituden),np.std(kinAmplituden)/np.sqrt(NoR),np.sqrt(np.sum(np.array(kinAmplitude)**2))/NoR,np.std(kinAmplitude)/NoR/2,np.sqrt(np.sum(np.array(kinAmplituden)**2))/NoR,np.std(kinAmplituden)/NoR/2,np.abs(np.sum(nmean)/NoR),np.sqrt(np.std(np.real(np.array(nmean)))**2+np.std(np.imag(np.array(nmean)))**2))/np.sqrt(NoR)],fmt="%s")
        noiseFile.close()
    plt.plot(time,Wges,label="Sum of energies")
    
    #plt.axhline(meanWtransfer[-1]*Wges[0],color="orange",linestyle="dashed")
    #plt.axhline((1-meanWtransfer[-1])*Wges[0],color="blue",linestyle="dashed")
    
    
    Linparams=opt.curve_fit(lin,time,Wges) #The growth of total energy
    plt.plot(time,lin(time,*Linparams[0]),"--g",label="energy grows with {}".format(Linparams[0][0]/Wges[0]*100)+r"% per $\omega_p^{-1}$")
    print("Absolute growth of total energy: {}%".format(Wges[-1]/Wges[0]*100))
    plt.legend()
    plt.savefig("PICEnergyPlot.pdf")
    
    ##--Plot the mean energy transfer for each run--##
    if(NoR>1 and plottransferevolution):
        plt.figure()
        plt.title("Mean Energy transfer vs. #Run")
        plt.plot(meanWtransfer)
        plt.plot(maxWtransfer)
    
    print("The mean value of energy transfer is:{}+-{}".format(np.mean(meanWtransfer),np.std(meanWtransfer)/np.sqrt(len(meanWtransfer))))
    print("The max value of energy transfer is:{}+-{}".format(np.mean(maxWtransfer),np.std(maxWtransfer)/np.sqrt(len(maxWtransfer))))
    print("The mean growth rate is:{}+-{}".format(np.mean(allgrowthrates),np.std(allgrowthrates)/np.sqrt(len(allgrowthrates))))
    
    ##--Plot the trajectory and velocity of the Drone Particles--##
    if NoDP>0:
        plt.figure()
        plt.title("position of a single particle")
        plt.plot(np.linspace(0,Nsteps,Nsteps+1)*deltat,xN1[:-1])
        plt.figure()
        plt.title("velocity of a single particle")
        plt.plot(np.linspace(0,Nsteps,Nsteps+1)*deltat,vN1[:-1])
    
    
    ##--Plot the temperatures of the two streams--##
    V1all=np.array(V1all); V2all=np.array(V2all)
    V1err=np.zeros((Nsteps+1));V2err=np.zeros((Nsteps+1))
    for i in range(Nsteps+1):
        V1err[i]=np.std(V1all[:,i])
        V2err[i]=np.std(V2all[:,i])
        
    if plottemperatureevolution:
        #def F3(x,a,b,c,r0):
        #    return (1/c*np.exp(-b*c*x)*(c*r0**b+a)-a/c)**(1/b)
        
        plt.figure()
        plt.title(r"Temperature evolution, $r_{T,0}$="+"{:.1f}".format(rT))
        plt.xlabel(r"$t\omega_p$")
        plt.ylabel(r"$V/V_{1,0}$")
        plt.plot(time,V1sum/NoR,label=r"$V_{1}$")
        plt.plot(time,V2sum/NoR,label=r"$V_{2}$")
        #V1param=opt.curve_fit(F3,time,V1sum/NoR)
        plt.plot(time,V2sum/V1sum,label="temperature ratio")
        #fitparam2
        #plt.plot(time,F3(time,*V1param[0]),"--k")
        plt.errorbar(time,V1sum/NoR,yerr=V1err/np.sqrt(NoR),color="blue")
        plt.errorbar(time,V2sum/NoR,yerr=V2err/np.sqrt(NoR),color="orange")
        plt.legend()
        plt.savefig("PICtemperaturePlot.pdf")
    
        print(rT,(V1sum[int(partcrossingtime/deltat)]/NoR-V1sum[0]/NoR)/partcrossingtime,(V2sum[int(partcrossingtime/deltat)]/NoR-V2sum[0]/NoR)/partcrossingtime,np.sqrt(V1err[int(partcrossingtime/deltat)]**2+V1err[0]**2)/partcrossingtime,np.sqrt(V2err[int(partcrossingtime/deltat)]**2+V2err[0]**2)/partcrossingtime)
    if write:
        rTFile.write("{} {} {} {} {}\n".format(rT,V1sum[int(partcrossingtime/deltat)]/NoR,V2sum[int(partcrossingtime/deltat)]/NoR,V1err[int(partcrossingtime/deltat)],V2err[int(partcrossingtime/deltat)]))
        #rTFile.write("{} {} {} {} {}\n".format(rT,(V1sum[int(partcrossingtime/deltat)]/NoR-V1sum[0]/NoR)/partcrossingtime,(V2sum[int(partcrossingtime/deltat)]/NoR-V2sum[0]/NoR)/partcrossingtime,np.sqrt(V1err[int(partcrossingtime/deltat)]**2+V1err[0]**2)/partcrossingtime,np.sqrt(V2err[int(partcrossingtime/deltat)]**2+V2err[0]**2)/partcrossingtime))
#plt.figure()
#plt.plot(1+np.linspace(0,rT-1,NoR),allgrowths1,"b.")
#plt.plot(1+np.linspace(0,rT-1,NoR),allgrowths2,".",color="orange")
if write:
    rTFile.close()
print("Total time:",realtime()-starttime)
