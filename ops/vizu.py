import os
import numpy as np
import matplotlib.pyplot as plt

def viz(data):
    x=[]
    y=[]
    rowx=[]
    rowy=[]
    '''
    while j<15:
        x=np.append(x,data[0][j])
        y=np.append(y,data[0][1])
        j+=1'''
    f,d=np.shape(data)
    print(f)
    for j in range(f):
        for i in range(d):
            if i<(d/2):
                x=np.append(x,data[j][i])
            else:
                y=np.append(y,data[j][i])
        rowx.append(x)
        rowy.append(y)
        x=[]
        y=[]

    '''Panoptics generate skeleton

    spinex,spiney,left_armx,left_army,left_legx,left_legy,right_armx,right_army,right_legx,right_legy=creat_skeleton(x,y)
    plot of data
    plt.figure(1)
    
    plt.plot(-spinex,-spiney, '-o') 
    plt.plot(-left_armx,-left_army, '-o')
    plt.plot(-left_legx,-left_legy, '-o')
    plt.plot(-right_armx,-right_army, '-o')
    plt.plot(-right_legx,-right_legy, '-o')'''
    plt.figure(1)
    plt.plot(rowx[0],-rowy[0],'o')
    plt.show()

def viz_gcn(data):
    j=0
    x=[]
    y=[]

    while j<15:
        x=np.append(x,data[j*2])
        y=np.append(y,data[j*2+1])
        j+=1
    '''Panoptics generate skeleton'''


    spinex,spiney,left_armx,left_army,left_legx,left_legy,right_armx,right_army,right_legx,right_legy=creat_skeleton(x,y)
    '''plot of data'''
    print('plot of data')
    plt.figure(1)
    plt.plot(-spinex,-spiney, '-o') 
    plt.plot(-left_armx,-left_army, '-o')
    plt.plot(-left_legx,-left_legy, '-o')
    plt.plot(-right_armx,-right_army, '-o')
    plt.plot(-right_legx,-right_legy, '-o')
    #plt.xlim([-0.5, 0.5])
    #plt.ylim([-0.5, 0.5])
    plt.show()

def creat_skeleton(x,y):
    spinex=[]
    spiney=[]
    left_armx=[]
    left_army=[]
    left_legx=[]
    left_legy=[]
    right_armx=[]
    right_army=[]
    right_legx=[]
    right_legy=[]
    spinex=np.append(spinex,x[1])
    spinex=np.append(spinex,x[0])
    spinex=np.append(spinex,x[2])


    spiney=np.append(spiney,y[1])
    spiney=np.append(spiney,y[0])
    spiney=np.append(spiney,y[2])

    left_armx=np.append(left_armx,x[0])
    left_armx=np.append(left_armx,x[3])
    left_armx=np.append(left_armx,x[4])
    left_armx=np.append(left_armx,x[5])

    left_army=np.append(left_army,y[0])
    left_army=np.append(left_army,y[3])
    left_army=np.append(left_army,y[4])
    left_army=np.append(left_army,y[5])

    left_legx=np.append(left_legx,x[2])
    left_legx=np.append(left_legx,x[6])
    left_legx=np.append(left_legx,x[7])
    left_legx=np.append(left_legx,x[8])

    left_legy=np.append(left_legy,y[2])
    left_legy=np.append(left_legy,y[6])
    left_legy=np.append(left_legy,y[7])
    left_legy=np.append(left_legy,y[8])

    right_armx=np.append(right_armx,x[0])
    right_armx=np.append(right_armx,x[9])
    right_armx=np.append(right_armx,x[10])
    right_armx=np.append(right_armx,x[11])

    right_army=np.append(right_army,y[0])
    right_army=np.append(right_army,y[9])
    right_army=np.append(right_army,y[10])
    right_army=np.append(right_army,y[11])

    right_legx=np.append(right_legx,x[2])
    right_legx=np.append(right_legx,x[12])
    right_legx=np.append(right_legx,x[13])
    right_legx=np.append(right_legx,x[14])

    right_legy=np.append(right_legy,y[2])
    right_legy=np.append(right_legy,y[12])
    right_legy=np.append(right_legy,y[13])
    right_legy=np.append(right_legy,y[14])

    return spinex,spiney,left_armx,left_army,left_legx,left_legy,right_armx,right_army,right_legx,right_legy

def viz_aberman(data,s,f):
    j=0
    n=0
    x=[]
    y=[]
    while j<64:
        if n<15:
            x=np.append(x,data[s][f][n][0])
            y=np.append(y,data[s][f][n][1])
        j+=1
        n+=1

    sx,sy,lax,lay,lgx,lgy,rax,ray,rgx,rgy=creat_skeleton(x,y)
    

    '''plot of data'''
    print('plot of data')
    plt.figure(1)
    plt.plot(-sx,-sy, '-o') 
    plt.plot(-lax,-lay, '-o')
    plt.plot(-lgx,-lgy, '-o')
    plt.plot(-rax,-ray, '-o')
    plt.plot(-rgx,-rgy, '-o')
    
    #plt.xlim([-5, 5])
    #plt.ylim([-5, 5])
    plt.show()
    
    return sx,sy,lax,lay,lgx,lgy,rax,ray,rgx,rgy

    
