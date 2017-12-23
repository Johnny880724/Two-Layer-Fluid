'''
Created on Oct 31, 2017

@author: Johnny Tsao
'''
from visual import*

runtime_max = 20
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
     
def initialize():
    #initialize density, height
    density[0:][0:][0:]=density1
    H1[0:][0:]=h1
    H2[0:][0:]=h2
    #set initial height    
    H1[0:5][0:3]=h1
    H2[0:5][0:3]=h2
    #set initial velocity
    ##vx[3][5][0:]=0
    ##vy[3][5][0:]=0
    ##vz[3][5][0:]=0
    pass


##partial velocity
def partial_vx_x(i,j,k):
    delta_xx=vx[i+1][j][k] - vx[i-1][j][k]
    return delta_xx/(2.*dx)

def partial_vy_x(i,j,k):
    delta_xy=vy[i+1][j][k] - vy[i-1][j][k]
    return delta_xy/(2*dx)

def partial_vz_x(i,j,k):
    delta_xz=vz[i+1][j][k] - vz[i-1][j][k]
    return delta_xz/(2*dx)

def partial_vx_y(i,j,k):
    delta_yx=vx[i][j+1][k] - vx[i][j-1][k]
    return delta_yx/(2*dy)

def partial_vy_y(i,j,k):
    delta_yy=vy[i][j+1][k] - vy[i][j-1][k]
    return delta_yy/(2*dy)

def partial_vz_y(i,j,k):
    delta_yz=vz[i][j+1][k] - vz[i][j-1][k]
    return delta_yz/(2*dy)

def partial_vx_z(i,j,k):
    delta_zx=vx[i][j][k+1] - vx[i][j][k-1]
    return delta_zx/(2*dz)

def partial_vy_z(i,j,k):
    delta_zy=vy[i][j][k+1] - vy[i][j][k-1]
    return delta_zy/(2*dz)

def partial_vz_z(i,j,k):
    delta_zz=vz[i][j][k+1] - vz[i][j][k-1]
    return delta_zz/(2*dz)

##laplace velocity
def laplace_vx(i,j,k):
    laplace_vx_x = vx[i+1][j][k] - 2*vx[i][j][k] + vx[i-1][j][k]
    laplace_vx_y = vx[i][j+1][k] - 2*vx[i][j][k] + vx[i][j-1][k]
    laplace_vx_z = vx[i][j][k+1] - 2*vx[i][j][k] + vx[i][j][k-1]
    
    return laplace_vx_x/dx**2 + laplace_vx_y/dy**2 + laplace_vx_z/dz**2

def laplace_vy(i,j,k):
    laplace_vy_x = vy[i+1][j][k] - 2*vy[i][j][k] + vy[i-1][j][k]
    laplace_vy_y = vy[i][j+1][k] - 2*vy[i][j][k] + vy[i][j-1][k]
    laplace_vy_z = vy[i][j][k+1] - 2*vy[i][j][k] + vy[i][j][k-1]
    
    return laplace_vy_x/dx**2 + laplace_vy_y/dy**2 + laplace_vy_z/dz**2

def laplace_vz(i,j,k):
    laplace_vz_x = vz[i+1][j][k] - 2*vz[i][j][k] + vz[i-1][j][k]
    laplace_vz_y = vz[i][j+1][k] - 2*vz[i][j][k] + vz[i][j-1][k]
    laplace_vz_z = vz[i][j][k+1] - 2*vz[i][j][k] + vz[i][j][k-1]
    
    return laplace_vz_x/dx**2 + laplace_vz_y/dy**2 + laplace_vz_z/dz**2
    
##partial height
def partial_h1_x(i,j):
    #partial_h1_x = H1[i][j] - H1[i-1][j] 
    #partial_h1_x = (max(H1[i][j], H1[i][j - 1]) - max(H1[i - 1][j], H1[i - 1][j -1]))
    partial_h1_x = ((H1[i][j] + H1[i][j - 1])/2 - (H1[i - 1][j] + H1[i - 1][j -1])/2)
    return partial_h1_x/dx

def partial_h1_y(i,j):
    #partial_h1_y = H1[i][j] - H1[i][j-1]
    #partial_h1_y = (max(H1[i][j], H1[i - 1][j]) - max(H1[i][j - 1], H1[i - 1][j - 1]))
    partial_h1_y = ((H1[i][j] + H1[i - 1][j])/2 - (H1[i][j - 1] + H1[i - 1][j - 1])/2)
    return partial_h1_y/dy

def partial_h2_x(i,j):
    #partial_h2_x = H2[i][j] - H2[i-1][j]
    #partial_h2_x = (max(H2[i][j], H2[i][j - 1]) - max(H2[i - 1][j], H2[i - 1][j -1]))
    partial_h2_x = ((H2[i][j] + H2[i][j - 1])/2 - (H2[i - 1][j] + H2[i - 1][j -1])/2)
    return partial_h2_x/dx

def partial_h2_y(i,j):
    #partial_h2_y = H2[i][j] - H2[i][j-1]
    #partial_h2_y = (max(H2[i][j], H2[i - 1][j]) - max(H2[i][j - 1], H2[i - 1][j - 1]))
    partial_h2_y = ((H2[i][j] + H2[i - 1][j])/2 - (H2[i][j - 1] + H2[i - 1][j - 1])/2)
    return partial_h2_y/dy

##Moving Function
def move1(i,j,k):
    delta_vx= -(vx[i][j][k]*partial_vx_x(i,j,k) + vy[i][j][k]*partial_vx_y(i,j,k) + vz[i][j][k]*partial_vx_z(i,j,k)) +\
              n/density1*laplace_vx(i,j,k) - g*(partial_h1_x(i,j) + (density2/density1)*partial_h2_x(i,j))
              
    delta_vy= -(vx[i][j][k]*partial_vy_x(i,j,k) + vy[i][j][k]*partial_vy_y(i,j,k) + vz[i][j][k]*partial_vy_z(i,j,k)) +\
              n/density1*laplace_vy(i,j,k) - g*(partial_h1_y(i,j) + (density2/density1)*partial_h2_y(i,j))
              
    delta_vz= -(vx[i][j][k]*partial_vz_x(i,j,k) + vy[i][j][k]*partial_vz_y(i,j,k) + vz[i][j][k]*partial_vz_z(i,j,k)) +\
              n/density1*laplace_vz(i,j,k)
              
    return array([delta_vx,delta_vy,delta_vz])

def move2(i,j,k):
    delta_vx= -(vx[i][j][k]*partial_vx_x(i,j,k) + vy[i][j][k]*partial_vx_y(i,j,k) + vz[i][j][k]*partial_vx_z(i,j,k)) +\
              n/density2*laplace_vx(i,j,k) - g*(partial_h2_x(i,j) + partial_h1_x(i,j))/dx
              
    delta_vy= -(vx[i][j][k]*partial_vy_x(i,j,k) + vy[i][j][k]*partial_vy_y(i,j,k) + vz[i][j][k]*partial_vy_z(i,j,k)) +\
              n/density2*laplace_vy(i,j,k) - g*(partial_h2_y(i,j) + partial_h1_y(i,j))/dy
              
    delta_vz= -(vx[i][j][k]*partial_vz_x(i,j,k) + vy[i][j][k]*partial_vz_y(i,j,k) + vz[i][j][k]*partial_vz_z(i,j,k)) +\
              n/density2*laplace_vz(i,j,k)
              
    return array([delta_vx,delta_vy,delta_vz])

##height increase function  
def delta_h1(i,j):
    delta_h=0
    n0 = int((H1[i][j] + H1[i-1][j] + H1[i][j-1] + H1[i-1][j-1])/4./dz)
    for k in range(n0):
        delta_h+=vx[i][j][k]*(dz/dx)*dt
        delta_h+=vy[i][j][k]*(dz/dy)*dt
    if (i+1 < Nx):
        n2 = int((H1[i+1][j] + H1[i][j] + H1[i+1][j-1] + H1[i][j-1])/4./dz)
        for k in range(n2):
            delta_h-=vx[i+1][j][k]*(dz/dx)*dt
            delta_h+=vy[i+1][j][k]*(dz/dy)*dt
    if (j+1 < Ny):
        n1 = int((H1[i][j+1] + H1[i-1][j+1] + H1[i][j] + H1[i-1][j])/4./dz)
        for k in range(n1):
            delta_h+=vx[i][j+1][k]*(dz/dx)*dt
            delta_h-=vy[i][j+1][k]*(dz/dy)*dt
    
    if(i+1 < Nx and j+1 < Ny):
        n3 = int((H1[i+1][j+1] + H1[i][j+1] + H1[i+1][j] + H1[i][j])/4./dz)
        for k in range(n3):
            delta_h-=vx[i+1][j+1][k]*(dz/dx)*dt
            delta_h-=vy[i+1][j+1][k]*(dz/dy)*dt
    
    return delta_h

def delta_h2(i,j):
    delta_h=0
    n0_s = int((H1[i][j] + H1[i-1][j] + H1[i][j-1] + H1[i-1][j-1])/4./dz)
    n0_e = int((H2[i][j] + H2[i-1][j] + H2[i][j-1] + H2[i-1][j-1]+\
                H1[i][j] + H1[i-1][j] + H1[i][j-1] + H1[i-1][j-1])/4./dz)
    for k in range(n0_s, n0_e+1):
        delta_h+=vx[i][j][k]*(dz/dx)*dt
        delta_h+=vy[i][j][k]*(dz/dy)*dt
    if (i+1 < Nx):
        n2_s = int((H1[i+1][j] + H1[i][j] + H1[i+1][j-1] + H1[i][j-1])/4./dz)
        n2_e = int((H2[i+1][j] + H2[i][j] + H2[i+1][j-1] + H2[i][j-1]+\
                    H1[i+1][j] + H1[i][j] + H1[i+1][j-1] + H1[i][j-1])/4./dz)
        for k in range(n2_s, n2_e+1):
            delta_h-=vx[i+1][j][k]*(dz/dx)*dt
            delta_h+=vy[i+1][j][k]*(dz/dy)*dt
    if (j+1 < Ny):
        n1_s = int((H1[i][j+1] + H1[i-1][j+1] + H1[i][j] + H1[i-1][j])/4./dz)
        n1_e = int((H2[i][j+1] + H2[i-1][j+1] + H2[i][j] + H2[i-1][j]+\
                    H1[i][j+1] + H1[i-1][j+1] + H1[i][j] + H1[i-1][j])/4./dz)
        for k in range(n1_s, n1_e+1):
            delta_h+=vx[i][j+1][k]*(dz/dx)*dt
            delta_h-=vy[i][j+1][k]*(dz/dy)*dt
    
    if(i+1 < Nx and j+1 < Ny):
        n3_s = int((H1[i+1][j+1] + H1[i][j+1] + H1[i+1][j] + H1[i][j])/4./dz)
        n3_e = int((H1[i+1][j+1] + H1[i][j+1] + H1[i+1][j] + H1[i][j]+\
                    H2[i+1][j+1] + H2[i][j+1] + H2[i+1][j] + H2[i][j])/4./dz)
        for k in range(n3_s, n3_e+1):
            delta_h-=vx[i+1][j+1][k]*(dz/dx)*dt
            delta_h-=vy[i+1][j+1][k]*(dz/dy)*dt
    
    return delta_h

##determine air fluid1 fluid2
def determining_layer(i,j,k):
    if (i>0 and i<Nx and j>0 and j<Ny):
        h1_av = (H1[i][j] + H1[i-1][j] + H1[i][j-1] + H1[i-1][j-1])/4.
        h2_av = (H2[i][j] + H2[i-1][j] + H2[i][j-1] + H2[i-1][j-1])/4.
        if (k*dz > h1_av + h2_av):
            return 3 ##air or fluid 3
           
        elif (k*dz > h1_av and k*dz < h1_av + h2_av):
            return 2 ##fluid2
        
        else:
            return 1 ##fluid1
    else:
        return 0 ##edge
    

##run program
def run():
    
    dt,t,timer,runtime=0.001,0,0,0
    while(runtime < runtime_max):
        rate(real_time_rate)
        t+=dt
        timer+=dt
        
        if timer>runtime_rate *dt:
            timer=0
            
            for i in range(Nx):
                for j in range(Ny):
                    fluid1_sto[i][j][runtime] = H1[i][j]
                    fluid2_sto[i][j][runtime] = H2[i][j]
            runtime+=1
            print "process: %3.1f" % round(float(runtime*1.0/runtime_max * 100),1), "%"
            #sys.stdout.write("\rDoing thing %i" % i)
            #sys.stdout.flush()
        H1[0:][0] = h1 - A*(sin(-omega*t))
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    layer = determining_layer(i,j,k)
                    if layer == 3 or layer == 0:                  
                        vx[i][j][k]=0
                        vy[i][j][k]=0
                        vz[i][j][k]=0
                        
                    elif layer == 2:
                        delta_v_get = dt*move2(i,j,k)
                        deltaV_x[i][j][k]=delta_v_get[0]
                        deltaV_y[i][j][k]=delta_v_get[1]
                        deltaV_z[i][j][k]=delta_v_get[2]
                        density[i][j][k] = density2
                    
                    elif layer == 1:
                        delta_v_get = dt*move1(i,j,k)
                        deltaV_x[i][j][k]=delta_v_get[0]
                        deltaV_y[i][j][k]=delta_v_get[1]
                        deltaV_z[i][j][k]=delta_v_get[2]
                        density[i][j][k] = density1
        
        
        vx[0:][0:][0:] += deltaV_x[0:][0:][0:]
        vy[0:][0:][0:] += deltaV_y[0:][0:][0:]
        vz[0:][0:][0:] += deltaV_z[0:][0:][0:]
        for i in range(Nx):
            for j in range(Ny):
                if(H1[i][j] > Nz*dz):
                    H1[i][j] = Nz*dz
                if(H2[i][j] + H1[i][j] > Nz*dz):
                    H2[i][j] = Nz * dz - H1[i][j]
                H1[i][j] += delta_h1(i,j)
                H2[i][j] += delta_h2(i,j)
                if(H1[i][j] < 0 or H2[i][j] < 0):
                    negative_height_error(i,j)

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
        rate(0.1/dt)
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
def collect():
    open_file = open('Two_layer_fluid.txt','w')
    for i in range(Nx):
        for j in range(Ny):
            for runtime in range(runtime_max):
                open_file.write(str(fluid1_sto[i][j][runtime])+"\n")
                open_file.write(str(fluid2_sto[i][j][runtime])+"\n")
    open_file.close()
    
if __name__ == '__main__':
    initialize()    
    run()
    print "finish"
    #collect()
    demo()
            
    
    
