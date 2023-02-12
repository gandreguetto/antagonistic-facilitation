'''

    This file contains the python code for the calculations used in the manuscript: "The evolutionary stability of antagonistic plant facilitation across environmental gradients"

'''


########## packages used ########## 

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import simps 
import seaborn as sns
import matplotlib

###################################

########## model parameters ##########

WUE = 1     # resource use efficiency
cr = 5      # cost of fine roots
ct1 = 0.2   # cost of transportation roots (in the manuscript it is simply ct but here we want to differentiate it from the opportunistic                     cost so that the opportunistic cost can be set to zero when we look for the evolutionarily stable strategies)
ct2 = 0.2   # cost of transportation roots
ce =  0.1   # cost of facilitation trais
delta = 0.1 # resource decay rate

params = {'a' : 1.,   # alpha (uptake rate)
          'w' : 5.,   # omega (potential resource input)
          'b' : 0.08} # proportion of omega abiotically available

###################################


############################### Remove Remove Remove RemoveRemove RemoveRemove RemoveRemove RemoveRemove RemoveRemove Remove
##############################

q = 1. # WUE2/WUE1 = WUE/WUE = 1
Cb1 = 5. # cr
Ct1 = 0.2 # ct
Cb2 = 5. # cr
Ct2 = 0. # ct
Ce = .1 # ce
d = 0.

params = {'a1' : 10., # alpha1/delta = alpha / delta
          'a2' : 10., # alpha2/delta = alpha / delta
          'w' : 5., # w * WUE 
          'b' : 0.08}
########################################################################################################################################


def root_densities_engineer(x, f):
    '''
    Solves the 7th order equation that gives the engineer spatial configuration of root densities 
    when interacting with a spreading opportunistic. The polynomial equation is obtained from 
    equations S1.19 and S1.20 from the Supplementary Material (SM). Returns the real root that
    gives the maximum net resource gain.

    Parameters:
    :param x: float giving the spatial position (l in the manuscript)
    :param f: float giving the mining trait (phi in the manuscript)
    '''

    ### defining new parameters to simplify the notation when writing the equations 
    c1 = cr + ce * f + ct1 * x**2
    c2 = cr + ct2 * (x - d) **2
    z1 = WUE * params['w'] / c1
    z2 = WUE * params['w'] / c2
    z = (z1 + z2) * params['a'] / delta

    ### the coefficients of the 7th order polynomial equation
    coeff = [z**2 * (params['a'] / delta)**2 * f**5,
             z**2 * (params['a'] / delta)**2 * f**4 + 2 * z * (params['a'] / delta) * f**4 * ((params['a'] / delta) * (1 + params['b']) * z + z2 * (params['a'] / delta) * f) - z2 * (params['a'] / delta) * f**5 * (params['a'] / delta)**3 * z1**2,
             2 * z * (params['a'] / delta) * f**3 * ((params['a'] / delta) * (1 + params['b']) * z + z2 * (params['a'] / delta) * f)+ f**3 * ((params['a'] / delta) * (1 + params['b']) * z + z2 * (params['a'] / delta) * f)**2 + 2 * z * (params['a'] / delta) * f**3 * ((params['a'] / delta) * params['b'] * z + z2 * (params['a'] / delta) * f * (1 + params['b'])) - z1**2 * (params['a'] / delta)**3 * f**4 * z2 * (params['a'] / delta) * params['b'] - z1**2 * (params['a'] / delta)**2 * f**5 * z2 * (params['a'] / delta) - 4 * z1**2 * (params['a'] / delta)**3 * f**4 * z2 * (params['a'] / delta),
             2 * z * (params['a'] / delta) * f**2 * ((params['a'] / delta) * params['b'] * z + z2 * (params['a'] / delta) * f * (1 + params['b'])) + 2 * z * (params['a'] / delta) * f**3 * z2 * (params['a'] / delta) * params['b'] + f**2 * ((params['a'] / delta) * (1 + params['b']) * z + z2 * (params['a'] / delta) * f)**2 + 2 * f**2 * ((params['a'] / delta) * (1 + params['b']) * z + z2 * (params['a'] / delta) * f) * ((params['a'] / delta) * params['b'] * z + z2 * (params['a'] / delta) * f * (1 + params['b'])) - z1**2 * (params['a'] / delta)**2 * f**4 * z2 * (params['a'] / delta) * params['b']  - 4 * z1**2 * (params['a'] / delta)**3 * f**3 * z2 * (params['a'] / delta) * params['b'] - 4 * z1**2 * (params['a'] / delta)**2 * f**4 * z2 * (params['a'] / delta)  - 4 * z1**2 * (params['a'] / delta)**3 * f**3 * z2 * (params['a'] / delta) - 2 * z1**2  * (params['a'] / delta)**3 * f**3 * params['b'] * z2 * (params['a'] / delta),
             2 * z * (params['a'] / delta) * f**2 * z2 * (params['a'] / delta) * params['b'] + 2 * f * ((params['a'] / delta) * (1 + params['b']) * z + z2 * (params['a'] / delta) * f) * ((params['a'] / delta) * params['b'] * z + z2 * (params['a'] / delta) * f * (1 + params['b'])) + 2 * f**2 * ((params['a'] / delta) * (1 + params['b']) * z + z2 * (params['a'] / delta) * f) * z2 * (params['a'] / delta) * params['b'] + f * ((params['a'] / delta) * params['b'] * z + z2 * (params['a'] / delta) * f * (1 + params['b']))**2 - 4 * z1**2 * (params['a'] / delta)**2 * f**3 * z2 * (params['a'] / delta) * params['b']  - 2 * z1**2 * (params['a'] / delta)**3 * f**2 * params['b']**2 * z2 * (params['a'] / delta)  - 2 * z1**2 * (params['a'] / delta)**2 * f**3 * params['b'] * z2 * (params['a'] / delta)  - 4 * z1**2 * (params['a'] / delta)**3 * f**2 * z2 * (params['a'] / delta) * params['b'] - 4 * z1**2 * (params['a'] / delta)**2 * f**3 * z2 * (params['a'] / delta) - 4 * z1**2 * (params['a'] / delta)**3 * f**2 * z2 * (params['a'] / delta) * params['b'],
             2 * f * ((params['a'] / delta) * (1 + params['b']) * z + z2 * (params['a'] / delta) * f) * z2 * (params['a'] / delta) * params['b'] + ((params['a'] / delta) * params['b'] * z + z2 * (params['a'] / delta) * f * (1 + params['b']))**2 + 2 * f * ((params['a'] / delta) * params['b'] * z + z2 * (params['a'] / delta) * f * (1 + params['b'])) * z2 * (params['a'] / delta) * params['b'] - 2 * z1**2 * (params['a'] / delta)**2 * f**2 * z2 * (params['a'] / delta) * params['b']**2  - 4 * z1**2 * (params['a'] / delta)**2 * f**2 * params['b'] * z2 * (params['a'] / delta)  - 4 * z1**2 * (params['a'] / delta)**3 * f * params['b']**2 * z2 * (params['a'] / delta)  - 4 * z1**2 * (params['a'] / delta)**2 * f**2 * z2 * (params['a'] / delta) * params['b'] - z1**2 * (params['a'] / delta)**3 * f * z2 * (params['a'] / delta) * params['b']**2,
             2 * ((params['a'] / delta) * params['b'] * z + z2 * (params['a'] / delta) * f * (1 + params['b'])) * z2 * (params['a'] / delta) * params['b'] + z2**2 * (params['a'] / delta)**2 * params['b']**2 * f - 4 * z1**2 * (params['a'] / delta)**2 * f * z2 * (params['a'] / delta) * params['b']**2  - z1**2 * (params['a'] / delta)**3 * params['b']**3 * z2 * (params['a'] / delta) - z1**2 * (params['a'] / delta)**2 * f * params['b']**2 * z2 * (params['a'] / delta),
             z2**2 * (params['a'] / delta)**2 * params['b']**2 - z1**2 * (params['a'] / delta)**2 * params['b']**2 * z2 * (params['a'] / delta) * params['b']
               ]

    dens = np.roots(coeff) # finds all roots
    density = 0
    g = 0
    pos_real = 0
    for roots in dens:
        if roots > 0 and np.imag(roots) == 0:
            dens2 = (np.sqrt(q * (params['a'] / delta) * params['w'] * (params['b'] + f * roots) * (1 + (params['a'] / delta) * roots) / ((cr + ct2 * (x - d)**2) * (1 + f * roots))) - 1 - (params['a'] / delta) * roots) / (params['a'] / delta)
            fit = fitness_two(x, f, roots, dens2)[0]
            if fit > 0:
                pos_real += 1
            if  fit > g:
                density = roots
                g = fit

    if pos_real > 1:
        print('There are {} positive real roots.'.format(pos_real))
        print(dens)
        print(params['b'])
        print(params['w'])
        print(f)
    return np.real(density)

def fitness_single(x, f, r1):
    '''
    Computes the net resource gain at x of a single engineer with density r1 and mining trait f

    :param x: float - spatial location
    :param f: float - mining trait
    :param r1: float - the density of a single engineer
    '''
    return WUE * params['w'] * (params['a'] / delta) * (params['b'] + f * r1) * r1 / ((1 + f * r1) * (1 + (params['a'] / delta) * r1)) - (cr + ce * f + ct1 * x**2) * r1

def roots_single(x, f):
    '''
    Finds the roots of the polynomial equation that gives the potential evolutionarily stable densities of a single engineer

    :param x: float - spatial location
    :param f: float - mining trait
    '''

    c1 = cr + ce * f + ct1 * x**2
    coeff = [(params['a'] / delta)**2 * f**2, 2 * (params['a'] / delta) * f * ((params['a'] / delta) + f), (1 + WUE * params['w'] * f * params['b'] / c1) * (params['a'] / delta)**2 + (4 - WUE * params['w'] * (params['a'] / delta) / c1) * (params['a'] / delta) * f + (1 - WUE * params['w'] * (params['a'] / delta) / c1) * f**2, 2 * ((params['a'] / delta) + (1 - WUE * params['w'] * (params['a'] / delta) / c1) * f), (1 - WUE * params['w'] * (params['a'] / delta) * params['b']/ c1)]

    return np.roots(coeff)

def root_density_single1(x, f):
    '''
    Finds the roots of the single engineer polymial equation and returns the root that results in the largest net resource gain

    :param x: float - spatial location
    :param f: float - mining trait
    '''
    
    dens = roots_single(x, f)
    density = 0
    g = 0
    for roots in dens:
        if roots > 0 and np.imag(roots) == 0:
            fit = fitness_single(x, f, roots) 
            if  fit > g:
                density = roots
                g = fit

    return density

def fitness_two(x, f, r1, r2):
    '''
    Given the densities r1 and r2 the net resource gains are returned 

    :param r1: float - engineer density 
    :param r2: float - opportunistic density
    :param x: float - spatial location
    :param f: float - mining trait
    '''

    g1 =  WUE * params['w'] * (params['a'] / delta) * (params['b'] + f * r1) * r1 / ((1 + f * r1) * (1 + (params['a'] / delta) * r1 + (params['a'] / delta) * r2)) - (cr + ce * f + ct1 * x**2) * r1
    g2 =  WUE * params['w'] * (params['a'] / delta) * (params['b'] + f * r1) * r2 / ((1 + f * r1) * (1 + (params['a'] / delta) * r1 + (params['a'] / delta) * r2)) - (cr + ct2 * (x - d)**2) * r2

    return g1, g2

def total_fitness1(x, f, r1, r2):
    '''
    Integrates over the space the net resource gain of the engineer, returning the total net gain

    :param r1: float - engineer density 
    :param r2: float - opportunistic density
    :param x: float - spatial location
    :param f: float - mining trait
    '''
    return simps(fitness_two(x, f, r1, r2)[0], x)

def total_fitness2(x, f, r1, r2):
    '''
    Integrates over the space the net resource gain of the opportunistic, returning the plant-level resource gain

    :param r1: float - engineer density 
    :param r2: float - opportunistic density
    :param x: float - spatial location
    :param f: float - mining trait
    '''

    return simps(fitness_two(x, f, r1, r2)[1], x)

def total_fitness1_phi():
    '''
    Computes the engineer plant-level resource gain for a range of mining traits.
    '''
    
    f = np.arange(0, 18, .1)

    y = []
    for p in f:
        r = []
        distributions = spatial_dist_root_dens(p) 
        x = distributions[0] 
        r1, r2 = distributions[3:5]

        tot = total_fitness1(x, p, r1, r2)
        y = np.append(y, tot)

    return f, y    

def root_density_single2(x):
    '''
    Calculates the evolutionarily stable density of a single opportunistic species at x.         
    '''
    
    c2 = cr + ct2 * (x - d)**2
    return max((np.sqrt(WUE * params['w'] * (params['a'] / delta) * params['b'] / c2) - 1)/(params['a'] / delta), 0)

def spatial_dist_root_dens(f):
    x = np.arange(0, 70, 0.1)    

    r1 = []
    r2 = []

    q1 = []
    q2 = []

    r1_single = []
    r2_single = []

    g1_two = []
    g2_two = []

    for i in range(len(x)):
        dens1_single = root_density_single1(x[i], f)
        dens2_single = root_density_single2(x[i])
        dens1 = root_densities_engineer(x[i], f)
        dens2 = (np.sqrt(WUE * (params['a'] / delta) * params['w'] * (params['b'] + f * dens1) * (1 + (params['a'] / delta) * dens1) / ((cr + ct2 * (x[i] - d)**2) * (1 + f * dens1))) - 1 - (params['a'] / delta) * dens1) / (params['a'] / delta)
        fit1_two, fit2_two = fitness_two(x[i], f, dens1, dens2)

        q1 = np.append(q1, dens1)
        q2 = np.append(q2, dens2)

        if min(dens1, dens2) > 0 and min(fit1_two, fit2_two) > 0:
            r1 = np.append(r1, dens1)
            r2 = np.append(r2, dens2)

        elif dens1 > 0 and fit1_two > 0:
            if dens1_single > 0 and fitness_two(x[i], f, dens1_single, 0)[0] > 0:
                r1 = np.append(r1, dens1_single)
                r2 = np.append(r2, 0.)
            else:
                r1 = np.append(r1, 0)
                r2 = np.append(r2, 0)

        elif dens2 > 0 and fit2_two > 0:
            if dens2_single > 0 and fitness_two(x[i], f, 0, dens2_single)[1] > 0:
                r1 = np.append(r1, 0.)
                r2 = np.append(r2, dens2_single)
            else:
                r1 = np.append(r1, -100.)
                r2 = np.append(r2, 100.)

        else:
            r1 = np.append(r1, 0) 
            r2 = np.append(r2, 0) 

            
        r1_single = np.append(r1_single, dens1_single)
        r2_single = np.append(r2_single, dens2_single)

        g1_two = np.append(g1_two, fit1_two)
        g2_two = np.append(g2_two, fit2_two)


    return x, r1_single, r2_single, r1, r2, g1_two, g2_two, q1, q2    

def plot_root_densities():
#    tf = total_fitness1_phi()
#    j = np.argmax(tf[1])
#    f = tf[0][j]
    f = 33.4
    print(f)
    
    x, r1_single, r2_single, r1, r2, g1_two, g2_two, q1, q2 = spatial_dist_root_dens(f)        

    plt.ion()

#    plt.plot(x, r1_single, color = 'b', linestyle = '--')
#    plt.plot(x, r2_single, color = 'orange', linestyle = '--')
#    plt.plot(-x, r1_single, color = 'b', linestyle = '--')
#    plt.plot(-x, r2_single, color = 'orange', linestyle = '--')

#    plt.plot(x, r1, color = 'b')
#    plt.plot(x, r2, color = 'orange')
#    plt.plot(-x, r1, color = 'b')
#    plt.plot(-x, r2, color = 'orange')
    
#    plt.plot(x, g1_two, color = 'c', label = '$G_1$')
#    plt.plot(x, g2_two, color = 'gray', label = '$G_2$')
#    plt.plot(-x, g1_two, color = 'c')
#    plt.plot(-x, g2_two, color = 'gray')

    plt.plot(x, q1, color = 'g', label = '$R_1$')
    plt.plot(x, q2, color = 'brown', label = '$R_2$')
    plt.plot(-x, q1, color = 'g')
    plt.plot(-x, q2, color = 'brown')
    
    plt.xlabel('$x$')
    plt.ylabel('$R_1, R_2$')
    plt.legend()
    
def total_fitness(x, f, r1, r2):
    g1, g2 = fitness_two(x, f, r1, r2)
    tot1 = simps(g1, x)
    tot2 = simps(g2, x)

    return tot1, tot2

def total_biomass(x, r):
    return simps(r, x)

def interaction_coeffs(f):
#    tf = total_fitness_phi()
#    j = np.argmax(tf[1])
#    f = tf[0][j]
#    f = 18.8
#    print(f)

    r = spatial_dist_root_dens(f)

    x_in = []
    r2_single_in = []
    r1_in = []
    r2_in = []

    for i in range(len(r[0])):
        if min(r[3][i], r[4][i]) > 0:
            x_in = np.append(x_in, r[0][i])
            r2_single_in = np.append(r2_single_in, r[2][i])
            r1_in = np.append(r1_in, r[3][i])
            r2_in = np.append(r2_in, r[4][i])

    if len(x_in) > 0 :        
        tot2_single = total_fitness(x_in, f, np.zeros(len(r2_single_in)), r2_single_in)[1]
        tot2 = total_fitness(x_in, f, r1_in, r2_in)[1]

        tot2_biomass_single = total_biomass(x_in, r2_single_in)
        tot2_biomass = total_biomass(x_in, r2_in)
#        print(x_in[0])
#        print(x_in[-1])
        return (tot2 - tot2_single)/(tot2 + tot2_single), (tot2_biomass - tot2_biomass_single)/(tot2_biomass + tot2_biomass_single)

    elif np.dot(r[4], r[4]) > 0:
        return 10,  10

    else:
        return -10,  -10
    
def bifur_b_w(phi):
#def bifur_b_w():
    params['ct2'] = 0.2

    x1 = np.arange(0, 1 + 0.02, 0.02)
    x2 = np.arange(0, 10 + 0.2, 0.2)

    yf = np.zeros((len(x2), len(x1)))
    ym = np.zeros((len(x2), len(x1)))
    p = np.zeros((len(x2), len(x1)))

    for j in range(len(x1)):
        for i in range(len(x2)):
            params['b'] = x1[j]
            params['w'] = x2[i]
            print(params['b'])
#            t = total_fitness1_phi()
#            f = t[0][np.argmax(t[1])]
            f = phi[i][j]
            yf[i][j], ym[i][j] = interaction_coeffs(f)
            p[i][j] = f 

    return yf, ym, p        
#    return p        

def bifur_b_a(phi):
#def bifur_b_a():
    params['ct2'] = 0.2

    x1 = np.arange(0, 1 + 0.02, 0.02)
    x2 = np.arange(0, 30 + 0.2, 0.6)

    yf = np.zeros((len(x2), len(x1)))
    ym = np.zeros((len(x2), len(x1)))
    p = np.zeros((len(x2), len(x1)))

    for j in range(len(x1)):
        for i in range(len(x2)):
            params['b'] = x1[j]
            params['a'] = x2[i]
            print(params['b'])
            #t = total_fitness1_phi()
            #f = t[0][np.argmax(t[1])]
            f = phi[i][j]
            yf[i][j], ym[i][j] = interaction_coeffs(f)
            p[i][j] = f
            
    return yf, ym, p        

def plot_heat_map_fit(y):
    num_xticks = 10
    num_yticks = 10
#    cmap = sns.color_palette("BrBG", as_cmap = True)
    cmap = sns.color_palette("RdBu", as_cmap = True)
    #    cmap = sns.color_palette("PiYG", as_cmap = True)
    cmap.set_under('black')
    cmap.set_over('yellow')
    ax = sns.heatmap(y, cmap=cmap, vmin = -1, vmax = 1, center = 0, xticklabels = num_xticks, yticklabels = num_yticks, cbar_kws={'label': '$y_f$', 'ticks':[-1, -.5, .0, 0.5, 1]}) 
    ax.invert_yaxis() 

#    ax.set_xticklabels(np.around(np.arange(0, 1 + 0.02, 0.02*num_xticks), 2))
#    ax.set_yticklabels(np.arange(0, 30 + 0.2, 0.6*num_yticks).astype(int))

    ax.set_xticklabels(np.around(np.arange(0, 1 + 0.02, 0.02*num_xticks), 2))
    ax.set_yticklabels(np.arange(0, 10 + 0.2, 0.2*num_yticks).astype(int))

#    ax.set_xticklabels(np.around(np.arange(0, 1, 0.05*num_xticks), 2))
#    ax.set_yticklabels(np.around(np.arange(0, 50, 2*num_yticks), 2))
    
    plt.xlabel('$b$')
#    plt.ylabel(r'$\delta^{-1}$')
    plt.ylabel(r'$\omega$')

def plot_heat_map_phi(y):
    num_xticks = 10
    num_yticks = 10
    ax = sns.heatmap(y, cmap='mako', vmin = 0., xticklabels = num_xticks, yticklabels = num_yticks, cbar_kws={'label': '$\phi$'})#, 'ticks':[0, 10, 20, 30, 40, 50]}) 
    ax.invert_yaxis() 

    ax.set_xticklabels(np.around(np.arange(0, 1 + 0.02, 0.02*num_xticks), 2))
    ax.set_yticklabels(np.around(np.arange(0, 30 + 0.2, 0.6*num_yticks), 2).astype(int))

#    ax.set_xticklabels(np.around(np.arange(0, 1, 0.05*num_xticks), 2))
#    ax.set_yticklabels(np.around(np.arange(0, 50, 2*num_yticks), 2))

    plt.xlabel('$b$')
    plt.ylabel(r'$\delta^{-1}$')

def bifur_b_G(f):
    b = np.arange(0, 1 + 0.005, 0.005)

    y1 = []
    y2 = []

    for i in range(len(b)):
        params['b'] = b[i]
        print(params['b'])

        r = spatial_dist_root_dens(f[i])
        x = r[0]
        r1 = r[3]
        r2 = r[4]

        tot = total_fitness(x, f[i], r1, r2)

        
        y1.append(tot[0])
        y2.append(tot[1])
            
    return b, y1, y2        

def bifur_b_phi():
    b = np.arange(0, 1 + 0.005, 0.005)

    params['ct2'] = 0.
    phi = []

    for bas in b:
        params['b'] = bas
        print(params['b'])
        t = total_fitness1_phi()
        f = t[0][np.argmax(t[1])]
        phi.append(f)
 
    return phi        
    

