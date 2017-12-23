'''
Created on Nov 30, 2017

@author: Johnny Tsao
'''
from visual import*

runtime_max = 1000
runtime_rate = 5
N=1

x,y,z=0.1,0.1,0.1
Nx,Ny,Nz=10*N,10*N,10*N
h1=0.4*z
h2=0.2*z
A=0.1*z
omega=2*pi/3*20
dx,dy,dz=x/Nx,y/Ny,z/Nz
density1=1000.0
density2=500.0
g=9.8
n=9E-1
points=zeros((Nx+1,Nz+1,4))
vx,vy,vz = zeros((Nx+1,Ny+1,Nz+1)),zeros((Nx+1,Ny+1,Nz+1)),zeros((Nx+1,Ny+1,Nz+1))
density = zeros((Nx+1,Ny+1,Nz+1))
deltaV_x = zeros((Nx+1,Ny+1,Nz+1))
deltaV_y = zeros((Nx+1,Ny+1,Nz+1))
deltaV_z = zeros((Nx+1,Ny+1,Nz+1))
H1=zeros((Nx,Ny),dtype = float)
H2=zeros((Nx,Ny),dtype = float)
start_width = Nx/6
dt,t,timer=0.001,0,0
real_time_rate=0.1/dt

fluid1_sto=zeros((Nx,Ny,runtime_max), dtype=float)
fluid2_sto=zeros((Nx,Ny,runtime_max), dtype=float)
pause=False
param=False
def keyinput(evt):
    global pause
    s = evt.key
    if (s == 'p'):
        pause = not pause

def negative_height_error(i,j):
    print("ERROR: height smaller than 0 at position",i,j)
    print("H1=",H1[i][j]," H2=",H2[i][j])
    while(True):
        sleep(1)
    pass

def demo():
    scene=display(title='Two layer fluid',height=1000,width=1000,center=(Nx*dx/2,Ny*dy/2,0),range=0.25,background=(0.8,0.9,0.8))
    scene.forward=vector(0,1,0)
    time=label(text='t=0',align='right',height=0.03,pos=vector(0.25,0,0.1),axis=vector(1,0,0),forward=vector(0,1,0), depth=-0.01, color=color.black)
    
    fluid1=empty(shape=(Nx,Ny), dtype=object)
    fluid2=empty(shape=(Nx,Ny), dtype=object)
    for i in range(Nx):
        for j in range(Ny):
            ##fluid1.append(box(opacity=0.4,h=h1,pos=vector((i+0.5)*dx,(j+0.5)*dy,h1/2),length=dx,height=dy,width=h1,color=(0.5,0.5,1)))
            ##fluid2.append(box(opacity=0.4,h=H2,pos=vector((i+0.5)*dx,(j+0.5)*dy,h1+h2/2),length=dx,height=dy,width=h2,color=(1,0.6,0.4)))
            fluid1[i][j] = (box(opacity=0.4,h=h1,pos=vector((i+0.5)*dx,(j+0.5)*dy,h1/2),length=dx,height=dy,width=h1,color=(0.5,0.5,1)))
            fluid2[i][j] = (box(opacity=0.1,h=h2,pos=vector((i+0.5)*dx,(j+0.5)*dy,h1+h2/2),length=dx,height=dy,width=h2,color=(1,0.6,0.4)))
    
    dt,t,timer,runtime=0.001,0,0,0
    while(runtime < runtime_max):
        rate(1)
        t+=dt
        timer+=dt
        if pause:
            while pause:
                param = True
                sleep(1)
            param = False
            print "pause"
        for i in range(Nx):
            for j in range(Ny):
                fluid1[i][j].pos=vector((i+0.5)*dx,(j+0.5)*dy,fluid1_sto[i][j][runtime]/2)
                fluid1[i][j].width=fluid1_sto[i][j][runtime]
                fluid2[i][j].pos=vector((i+0.5)*dx,(j+0.5)*dy,fluid1_sto[i][j][runtime]+fluid2_sto[i][j][runtime]/2)
                fluid2[i][j].width=fluid2_sto[i][j][runtime]
        if timer>=runtime_rate*dt:
            timer=0
            time.text="t=%.3f(s)" %round(float(t),3)
            runtime+=1
def getdata():
    open_file = open('Two_layer_fluid.txt','r')
    for i in range(Nx):
        for j in range(Ny):
            for runtime in range(runtime_max):
                fluid1_sto[i][j][runtime] = open_file.readline()
                fluid2_sto[i][j][runtime] = open_file.readline()
    open_file.close()

    
if __name__ == '__main__':
    getdata()
    demo()
            
    
    
