from NeuroTools import stgen
import numpy as np
from matplotlib import pyplot as plt

from scipy import stats

import seaborn as sns

import numpy as np
from scipy import stats
import scipy
import itertools

try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass


A = 1.0
x_coords,y_coords = np.meshgrid(np.arange(0.0,1.0,0.02),np.arange(0.0,1.0,0.02))

r_0 = 0.5  #np.arange(0.5,10.0,9.5/N_filters)
r_max = 20.0

class RF_network(object):
    def __init__(self, N_filters):
        self.N_filters = N_filters
    def setVar(self, var):
        for key, value in var.items():
            setattr(self, key, value)


def get_selectivity(W,N_orientations=4,reward_ID=0):
    N_neurons = W.shape[0]
    selectivity = np.zeros(N_neurons)
    reward_selectivity = np.zeros(N_neurons)
    N_per_orientation = int(N_neurons/N_orientations)
    reward_idx = range(reward_ID*N_per_orientation,(reward_ID+1)*N_per_orientation)
    for i in xrange(N_orientations):
        same_idx = range(i*N_per_orientation,(i+1)*N_per_orientation)
        other_idx = list(set(range(N_neurons))-set(same_idx))
        for j in xrange(N_per_orientation):
            selectivity[i*N_per_orientation + j] = ((N_orientations-1)*np.mean(W[i*N_per_orientation + j,same_idx])-np.mean(W[i*N_per_orientation + j,other_idx]))
            if reward_ID == i:
                reward_selectivity[i*N_per_orientation + j] = 0.0
            else:
                reward_selectivity[i*N_per_orientation + j] = ((N_orientations-1)*np.mean(W[i*N_per_orientation + j,reward_idx])-np.mean(W[i*N_per_orientation + j,other_idx]))
    return selectivity, reward_selectivity

def measure_selectivity_from_frozen(net,sim_pars,W_plastic,W_IE):
    r_0 = 0.5
    r_max = 20.0

    dt = 0.1

    x_inh = 0.0
    W_EI = sim_pars['W_EI']

    sim_pars['inh_target'] = r_max*0.5

    presented_bar = np.zeros(int(T/sim_pars['stim_period']))
    responses = np.zeros((net.N_filters,len(net.theta_list)))

    for stim_idx in xrange(len(net.theta_list)):
        image_bar = generate_bars_image(net.theta_list[stim_idx], np.pi)
        x_corrs = np.zeros((net.N_filters,1))
        for idx in xrange(net.N_filters):
            x_corrs[idx] = 1.0*corr2(net.filters[idx][0],image_bar) + sim_pars['x_corr_mean']
        x_inh = 0.0
        x = np.zeros((net.N_filters,1))
        for t_idx in sim_pars['stim_period']:
            x += dt*(-1*x + get_rates(sim_pars['W_input']*x_corrs + 1.0*np.dot(np.transpose(W_plastic),x) - np.reshape(np.dot(x_inh,W_IE),(net.N_filters,1))) )
            x[x<0] = 0
            x_inh += dt*(-1*x_inh + get_rate(W_EI*np.sum(x)))
        responses[:,stim_idx] = x.flatten()

    return responses

def get_group_selectivity(W,group_idx):
    N_filters = W.shape[0]
    nongroup_idx = list(set(range(N_filters)) - set(group_idx))
    W_group_row = W[group_idx,:]
    W_nongroup_row = W[nongroup_idx,:]

    #return np.mean(W_group_row[:,group_idx])-np.mean(W_nongroup_row[:,nongroup_idx])
    return np.mean(W[group_idx,:],axis=0)-np.mean(W[nongroup_idx,:],axis=0)

def gabor_function(A,theta,phi,f,sigma_x,sigma_y,c_x,c_y,x_coords,y_coords):
    x_p = (x_coords-c_x)*np.cos(theta)-(y_coords-c_y)*np.sin(theta)
    y_p = (x_coords-c_x)*np.sin(theta)+(y_coords-c_y)*np.cos(theta)
    G = A*np.exp(-1*(x_p**2)/(2*sigma_x**2) -  (y_p**2)/(2*sigma_y**2) )*np.cos(2*np.pi*f*x_p + phi)
    return G

def corr2_coeff(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);

    # Finally get corr coeff
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))

def partial_corr(x,y,z,plot=True,color='b',xlabel='x',ylabel='y ',title='network'):
    beta_i = stats.linregress(z, x)
    beta_j = stats.linregress(z, y)

    line_x = beta_i[1] + np.multiply(beta_i[0],z)
    line_y =  beta_j[1] + np.multiply(beta_j[0],z)

    #print beta_i,beta_j

    #plt.scatter(x,line_x)
    #plt.scatter(y,line_y,color='r')

    res_j = np.subtract(y,line_y)
    res_i = np.subtract(x,line_x)

    corr = stats.spearmanr(res_i, res_j)
    print('partial correlation')
    print(corr)

    if plot:
        fig,axes=plt.subplots()
        plt.scatter(res_j,res_i,color=color)
        plt.title('Residuals ,' + title + ' ; ' + str(corr[0])+ ' ' + str(corr[1]))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    return corr


def generate_correlation_map(x, y):
    """Correlate each n with each m.

    Parameters
    ----------
    x : np.array
      Shape N X T.

    y : np.array
      Shape M X T.

    Returns
    -------
    np.array
      N X M array in which each element is a correlation coefficient.

    """
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must ' +
                         'have the same number of timepoints.')
    s_x = x.std(1, ddof=n - 1)
    s_y = y.std(1, ddof=n - 1)
    cov = np.dot(x,
                 y.T) - n * np.dot(mu_x[:, np.newaxis],
                                  mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])

import math

def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum());
    return r

def generate_bars_image(theta,phi,f=2,size=50,c_x=0.5,c_y=0.5):
    imag = gabor_function(A,theta,phi,f,5.0,5.0,c_x,c_y,x_coords,y_coords)
    imag[imag>A*0.5] = 1.0
    imag[imag<A*0.5] = 0.0

    return imag


def get_rates(x):
    #return (1+np.tanh(1*x))#+np.random.uniform(0,0.5,N_filters))
    x[x<=0] = r_0*np.tanh(x[x<=0]/r_0)
    x[x>0] = (r_max-r_0)*np.tanh(x[x>0]/(r_max-r_0))
    return x


def get_rate(x):
    #return (1+np.tanh(1*x))#+np.random.uniform(0,0.5,N_filters))
    if x < 0:
        x = r_0*np.tanh(x/r_0)
    else:
        x= (r_max-r_0)*np.tanh(x/(r_max-r_0))
    return x


def generate_RF_net(sim_pars,N_filters=200,N_per_group=50,pop_coupling_add_factor=0.05,pop_coupling_mult_factor=1.0,group_corr_size=50):
    results = []
    kernel_params = []

    theta_list = np.linspace(0.0,np.pi-np.pi/10,int(N_filters/N_per_group))

    for i in xrange(int(N_filters/N_per_group)):
        s = 2.0 #np.random.uniform(1,4)
        frequency = 2.0 #np.random.uniform(1.25*s)
        sigma = 2.0#0.2/s
        c_x = 0.5 # np.random.uniform(0.35,0.65)
        c_y = 0.5 # np.random.uniform(0.35,0.65)
        #kernel = gabor_kernel(frequency, theta=theta,offset=phi,sigma_x=sigma,sigma_y=sigma)

            # Save kernel and the power image for each image
        #results.append((kernel, [power(img, kernel) for img in images]))
        for j in xrange(N_per_group):
            #theta = np.random.uniform(0,2*np.pi) #
            if sim_pars['random_theta']:
                theta = np.random.uniform(0,2*np.pi) #
            else:
                theta = theta_list[i] + np.random.uniform(0,sim_pars['theta_jitter'])
            if sim_pars['random_phi']:
                phi = np.random.uniform(0,2*np.pi)
            else:
                phi = np.pi
            kernel = gabor_function(A,theta,phi,frequency,sigma,sigma,c_x,c_y,x_coords,y_coords)
            params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
            kernel_params.append(params)
            results.append((kernel, []))

    W = np.zeros((N_filters,N_filters))
    p_c = np.zeros((N_filters,N_filters))

    corrs = []

    for i,j in itertools.combinations(range(N_filters),2):
        #c = np.mean(signal.correlate2d(np.real(results[i][0]),np.real(results[j][0])))
        c = corr2((results[i][0]),(results[j][0]))
        #c = np.corrcoef(np.real(results[i][0]),np.real(results[j][0]))
        corrs.append(c)
        #c*= 1.0/5e-11
        p_c[i,j] = 0.55*c*c +0.22*c +0.064
        p_c[j,i] = p_c[i,j]

        if p_c[i,j] > np.random.uniform(0,1) or True:
            W[i,j] = 0.28*np.exp(2.5*c)
        #else:
        #    W[i,j] = np.random.uniform(0,0.2)
        if p_c[j,i] > np.random.uniform(0,1) or True:
            W[j,i] = 0.28*np.exp(2.5*c)


    if sim_pars['pc_dist'] == 'gamma':
        pop_coupling = np.random.gamma(4.0,1.0,N_filters)
    elif sim_pars['pc_dist'] == 'bimodal':
        pop_coupling = (np.random.binomial(1,sim_pars['pc_bimodal_prob'],N_filters)*sim_pars['alpha_range'])+1.0
    elif sim_pars['pc_dist'] == 'uniform':
        pop_coupling = np.random.uniform(1.0,sim_pars['alpha_range'],N_filters)
    elif sim_pars['pc_dist'] == 'lognormal':
        pop_coupling = np.random.lognormal(1,1.5,N_filters)
    elif sim_pars['pc_dist'] == 'normal':
        pop_coupling = np.abs(np.random.normal(20,10,N_filters))
    print(np.mean(pop_coupling))

    W = np.zeros((N_filters,N_filters)) # W[pre,post]

    p_c = np.zeros((N_filters,N_filters))

    corrs = []

    for i,j in itertools.combinations(range(N_filters),2):
        c = corr2((results[i][0]),(results[j][0]))
        corrs.append(c)
        p_c[i,j] = pop_coupling_mult_factor*(0.55*c*c +0.22*c +0.064) #+ pop_coupling_add_factor*pop_coupling[j]
        p_c[j,i] = pop_coupling_mult_factor*(0.55*c*c +0.22*c +0.064) #+ pop_coupling_add_factor*pop_coupling[i]
        if p_c[i,j] > np.random.uniform(0,1.0):
            W[i,j] = 0.28*np.exp(2.5*c)
        elif pop_coupling_add_factor*pop_coupling[j] > np.random.uniform(0,1.0):
            #W[i,j] = 0.28*np.exp(2.5*np.random.uniform(-1.0,1.0))
            W[i,j] = 0.28*np.exp(2.5*np.random.normal(0,0.2))
        #if pop_coupling_mult_factor*(p_c[j,i] + pop_coupling_add_factor*pop_coupling[i]) > np.random.uniform(0,1.0):
        if p_c[j,i] > np.random.uniform(0,1.0):
            W[j,i] = 0.28*np.exp(2.5*c)
        elif pop_coupling_add_factor*pop_coupling[i] > np.random.uniform(0,1.0):
            #W[j,i] = 0.28*np.exp(2.5*np.random.uniform(-1.0,1.0))
            W[j,i] = 0.28*np.exp(2.5*np.random.normal(0,0.2))

    group_corr = np.zeros((N_filters,N_filters))
    group_id = np.random.randint(0,N_filters,group_corr_size)

    for i,j in itertools.combinations(group_id,2):
        group_corr[i,j] = 1.0
        group_corr[j,i] = 1.0

    network = RF_network(N_filters)
    if sim_pars['uniform_prob_conn']:
        network.W_conn = np.random.binomial(1,sim_pars['W_density'],(N_filters,N_filters))
    else:
        network.W_conn = W>0
    if sim_pars['uniform_W_init']:
        network.W= sim_pars['W_scaling_target']*(1.0/N_filters)*np.ones((N_filters,N_filters))
        network.W[network.W_conn==0] = 0.0
    else:
        network.W = W.copy()
    network.pop_coupling = pop_coupling
    network.filters = results
    network.group_id = group_id
    network.group_corr = group_corr
    network.theta_list = theta_list

    if sim_pars.has_key('net_pars'):
        network.setVar(sim_pars['net_pars'])

    return network

def run_RF_net(net,sim_pars,coupling_dependence='corr',T=2000,sample_res=10,freeze_weights=False):
    W_plastic = net.W.copy()

    x = np.zeros((net.N_filters,1))
    x_corrs = np.zeros((net.N_filters,1))

    x_t = np.zeros((int(T/sample_res),net.N_filters))
    x_inh_t = np.zeros(int(T/sample_res))
    if not freeze_weights:
        W_t = np.zeros((int(T/sample_res),net.N_filters,net.N_filters))
        W_IE_t = np.zeros((int(T/sample_res),net.N_filters))

    if coupling_dependence == 'corr':
        eta1 = net.pop_coupling*sim_pars['eta1_base']
    elif coupling_dependence == 'anticorr':
        eta1 = (1.0/net.pop_coupling)*sim_pars['eta1_base']*np.mean(net.pop_coupling)/np.mean(1.0/net.pop_coupling)
    elif coupling_dependence == 'uncorr':
        eta1 = net.pop_coupling*sim_pars['eta1_base']
        np.random.shuffle(eta1)
    elif coupling_dependence == 'uniform':
        eta1 = sim_pars['eta1_base']*sim_pars['homogenous_base']

    if sim_pars['heterogenous_r_0']:
        r_0 = np.arange(0.5,sim_pars['r_0_range'],(sim_pars['r_0_range']-0.5)/net.N_filters)
    else:
        r_0 = 0.5
    r_max = 20.0

    dt = 0.1

    from NeuroTools import stgen

    stgen_drive = stgen.StGen()
    OU_drive = stgen_drive.OU_generator(1.,50,sim_pars['OU_global'],0.,0.,T).signal

    ext_OU = np.zeros((net.N_filters,T))
    for n_idx in xrange(net.N_filters):
        if sim_pars['heterogenous_ext_OU_tau']:
            ext_OU[n_idx] = stgen_drive.OU_generator(1.0,np.random.uniform(0.5,1.5)*sim_pars['ext_OU_tau'],1.0*sim_pars['ext_OU_sigma'],0,0,T,True)[0]
        elif sim_pars['heterogenous_ext_OU_sigma']:
            ext_OU[n_idx] = stgen_drive.OU_generator(1.0,sim_pars['ext_OU_tau'],np.random.uniform(0.5,1.5)*sim_pars['ext_OU_sigma'],0,0,T,True)[0]
        else:
            ext_OU[n_idx] = stgen_drive.OU_generator(1.0,sim_pars['ext_OU_tau'],1.0*sim_pars['ext_OU_sigma'],0,0,T,True)[0]

    x_inh = 0.0
    W_EI = sim_pars['W_EI']

    try:
        W_IE = net.W_IE.copy()
    except:
        W_IE = sim_pars['W_IE_init']*np.ones((net.N_filters,1))

    sim_pars['inh_target'] = r_max*0.5

    W_conn = net.W_conn.copy()

    presented_bar = np.zeros(int(T/sim_pars['stim_period']))

    if sim_pars['measure_responses']:
        presented_bar_response = np.zeros((int(T/sim_pars['stim_period']),net.N_filters))
        temp_response = np.zeros((sim_pars['stim_period'],net.N_filters))

    for t_idx in xrange(T):
        if t_idx%sim_pars['stim_period'] == 0:
            bar_orient_idx = np.random.randint(len(net.theta_list)) #np.random.uniform(0,2*np.pi)
            if sim_pars['present_random_bars']:
                image_bar = generate_bars_image(np.random.uniform(0,2*np.pi), np.random.uniform(0,2.0*np.pi))
            else:
                image_bar = generate_bars_image(net.theta_list[bar_orient_idx], np.pi)

            presented_bar[int(t_idx/sim_pars['stim_period'])] = bar_orient_idx

            if sim_pars['measure_responses']:
                temp_response = np.zeros((sim_pars['stim_period'],net.N_filters))

            for idx in xrange(net.N_filters):
                x_corrs[idx] = 1.0*corr2(net.filters[idx][0],image_bar) + np.random.normal(sim_pars['x_corr_mean'],sim_pars['x_corr_std']),#(50,50))
                #x_corrs = np.zeros(net.N_filters)
                #x_corrs[bar_orient_idx*10:(bar_orient_idx+1)*10] = 1.0

        #print 'x_corrs ' , x_corrs[15:30]
        x += dt*(-1*x + get_rates(sim_pars['W_input']*x_corrs + 1.0*np.dot(np.transpose(W_plastic),x) + OU_drive[t_idx] + ext_OU[:,t_idx].reshape(net.N_filters,1) - np.reshape(np.dot(x_inh,W_IE),(net.N_filters,1))) )
        #print 'OU_drive ', ext_OU[15:30,t_idx]
        #print 'x ', x[15:30]
        x[x<0] = 0
        x_inh += dt*(-1*x_inh + get_rate(W_EI*np.sum(x)))
        #print 'x inh ', x_inh

        if sim_pars['measure_responses']:
            temp_response[t_idx%sim_pars['stim_period']] = x.flatten()
            if (t_idx+1)%sim_pars['stim_period'] == 0:
                presented_bar_response[int((t_idx+1)/sim_pars['stim_period'])-1] = np.mean(temp_response,axis=0)
                #print presented_bar_response[:,:10]

        if not freeze_weights:
            W_IE += sim_pars['eta']*(x_inh*(x-sim_pars['inh_target']))

            dW_plastic = eta1*x*np.transpose(x)
            dW_plastic += sim_pars['scaling_rate']*(sim_pars['W_scaling_target'] - np.sum(W_plastic,axis=0))

            W_plastic[W_conn==1] += dW_plastic[W_conn==1]
            W_plastic[W_plastic<0] = 0
            W_plastic[W_plastic>sim_pars['W_max']] = sim_pars['W_max']

            np.fill_diagonal(W_plastic,0)

            if t_idx%sim_pars['prune_freq'] and sim_pars['prune_weights'] and sim_pars['prune_stop']>t_idx and sim_pars['prune_start']<t_idx:
                #W_conn[W_plastic<sim_pars['prune_thresh']] = 0
                W_pruned_temp = W_plastic<sim_pars['prune_thresh']
                W_pruned_temp[W_pruned_temp==1] = np.random.binomial(1,sim_pars['prune_prob'],W_pruned_temp[W_pruned_temp==1].shape)
                W_conn[W_pruned_temp==1] = 0
                W_plastic[W_conn==0] = 0

        if t_idx%sample_res == 0:
            #print 'sample_point'
            if not freeze_weights:
                W_t[int(t_idx/sample_res)] = W_plastic.copy()
                W_IE_t[int(t_idx/sample_res)] = W_IE.flatten().copy()
            x_t[int(t_idx/sample_res)] = x.reshape(net.N_filters)
            x_inh_t[int(t_idx/sample_res)] = x_inh

    run_results = {
        'W_plastic': W_plastic,
        'W_conn': W_conn,
        'x_t': x_t,
        'x_inh_t': x_inh_t,
        'T': T,
        't_res': sample_res,
        'eta1': eta1
    }

    if sim_pars['measure_responses']:
        run_results['presented_bar'] = presented_bar
        run_results['presented_bar_response'] = presented_bar_response

    if not freeze_weights:
        run_results['W_t'] = W_t
        run_results['W_IE_t'] = W_IE_t

    return run_results




def run_RF_net_old(net,coupling_dependence='corr',T=2000,group_corr_strength=2.5,t_res=100):

    W_plastic = net.W.copy()

    #W_plastic = np.ones((N_filters,N_filters))

    x = np.zeros((net.N_filters,1))
    x_corrs = np.zeros((net.N_filters,1))

    W_t = np.zeros((int(T/t_res),net.N_filters,net.N_filters))
    x_t = np.zeros((int(T/t_res),net.N_filters))

    eta1_base = 1.0*0.0001
    if coupling_dependence == 'corr':
        eta1 = net.pop_coupling*eta1_base
    elif coupling_dependence == 'anticorr':
        eta1 = (1.0/net.pop_coupling)*eta1_base*np.mean(net.pop_coupling)/np.mean(1/net.pop_coupling)
    elif coupling_dependence == 'uncorr':
        eta1 = eta1_base*np.mean(net.pop_coupling)

    eta1*=1.0

    eta2 = eta1_base*1.0

    alpha = 0.4
    beta = 0.4
    gamma = 0.45

    W_scaling_target = 20.0
    scaling_rate = 2.0*eta1_base


    W_max = 20.0

    stim_period = 100
    dt = 0.1

    from NeuroTools import stgen

    ext_OU_tau = 50.0
    ext_OU_sigma = 1.0

    stgen_drive = stgen.StGen()
    OU_drive = stgen_drive.OU_generator(1.,50,0.0,0.,0.,T).signal


    ext_OU = np.zeros((net.N_filters,T))
    for n_idx in xrange(net.N_filters):
        ext_OU[n_idx] = stgen_drive.OU_generator(1.0,ext_OU_tau,1.0*ext_OU_sigma,0,0,T,True)[0]

    x_inh = 0.0
    W_EI = 0.1
    W_IE = 0.1*np.ones((net.N_filters,1))

    eta = 0
    inh_target = r_max*0.5

    W_conn = net.W>0
    W_input = 0.5

    group_corr_frac_active = 1.0

    presented_bar = np.zeros(int(T/stim_period))
    presented_paired = np.zeros(int(T/stim_period))

    for t_idx in xrange(T):

        if t_idx%stim_period == 0:
            bar_orient_idx = np.random.randint(len(net.theta_list)) #np.random.uniform(0,2*np.pi)
            image_bar = generate_bars_image(np.random.uniform(net.theta_list[bar_orient_idx]), np.random.uniform(0,0.2))
#image_bar = image_bars[np.random.randint(len(image_bars))]  + image_bars[np.random.randint(len(image_bars))] + image_bars[np.random.randint(len(image_bars))]
            if np.random.uniform()>1.75:
                image_bar += generate_bars_image(np.pi/2, np.random.uniform(0.1,0.2))
                presented_paired[int(t_idx/stim_period)] = 5
            #    image_bar = generate_bars_image(np.pi/2,np.random.uniform(0.0,2*np.pi)) + generate_bars_image(3*np.pi/2,np.random.uniform(0.0,2*np.pi))
            presented_bar[int(t_idx/stim_period)] = bar_orient_idx

            group_active = np.random.choice(net.group_id,group_corr_frac_active*len(net.group_id),replace=False)

            for idx in xrange(net.N_filters):
                x_corrs[idx] = 1.0*corr2(net.filters[idx][0],image_bar) + np.random.normal(0.2,0.25),#(50,50))
                if idx in group_active:
                    x_corrs[idx] += group_corr_strength
                #x_corrs[idx] = np.random.normal(0.2,0.25)


        x += dt*(-1*x + get_rates(W_input*x_corrs + 1.0*np.dot(np.transpose(W_plastic),x) + OU_drive[t_idx] + ext_OU[:,t_idx].reshape(net.N_filters,1) - np.reshape(np.dot(x_inh,W_IE),(net.N_filters,1))) )
        #x = x_corrs + 0.1*np.dot(W_plastic,x_corrs)
        #x += np.random.normal(0.0,0.25)
        #x = get_rates(x)
        x[x<0] = 0

        x_inh += dt*(-1*x_inh + get_rate(W_EI*np.sum(x)))

        W_IE += eta*(x_inh*(x-inh_target))

        #W_plastic += alpha*x*np.transpose(x)

        dW_plastic = eta1*np.power(x,alpha)*np.power(W_plastic,beta)*np.power(x.transpose(),gamma) - eta2*W_plastic  + stats.norm.rvs(scale=0.1+W_plastic*0.0,size=(net.N_filters,net.N_filters))

        dW_plastic += scaling_rate*(W_scaling_target - np.sum(W_plastic,axis=0))

        W_plastic[net.W_conn] += dW_plastic[net.W_conn]
        W_plastic[W_plastic<0] = 0
        W_plastic[W_plastic>W_max] = W_max

        np.fill_diagonal(W_plastic,0)

        if t_idx%t_res == 0:
            W_t[int(t_idx/t_res)] = W_plastic.copy()
            x_t[int(t_idx/t_res)] = x.reshape(net.N_filters)

    run_results = {
        'W_plastic': W_plastic,
        'W_t': W_t,
        'x_t': x_t,
        'T': T,
        't_res': t_res
    }

    return run_results


def plot_RF_net_run(run_results,net):
    W_t = run_results['W_t']
    W_plastic = run_results['W_plastic']
    x_t = run_results['x_t']
    T = run_results['T']
    t_res = run_results['t_res']

#    plt.plot(W_t[:,:20,3],'g')
#    plt.plot(W_t[:,:20,2],'r')
#    plt.plot(W_t[:,:20,9],'b')
#    plt.show()
#
#    plt.plot(x_t[:,8:12])
#    #plt.xlim(400,1500)
#    #plt.plot(np.mean(x_t[:],axis=1),'black',lw=2)
#    plt.show()
#
#    #plt.hist([x_t[:,group_id],x_t[:,:20]],label=['over','under'])
#    #plt.legend()
#    #plt.show()
#
#    plt.hist([W_plastic[net.group_corr>0.05].flatten(),W_plastic[net.group_corr==0].flatten()],20,normed=True,range=(0.001,20),label=['over','under'],histtype='step')
#    plt.legend()
#    plt.show()
#
    mask_group = np.invert(np.logical_and(net.group_corr>0.05,W_plastic>0))
    mask_nongroup = np.invert(np.logical_and(net.group_corr<0.05,W_plastic>0))
    W_group = np.ma.array(W_plastic,mask=mask_group)
    W_nongroup = np.ma.array(W_plastic,mask=mask_nongroup)

    print('group',W_group.mean(), W_group.std())
    print('nongroup',W_nongroup.mean(), W_nongroup.std())

    #plt.pcolor(group_corr)
    #plt.show()

    selectivities_t = np.zeros((int(T/t_res),net.N_filters))
    group_selectivities_t= np.zeros((int(T/t_res),net.N_filters))

    control_group = np.random.randint(0,net.N_filters,50)

    for t_idx in xrange(int(T/t_res)):
        selectivities_t[t_idx] = get_selectivity(np.transpose(W_t[t_idx]),10,5)[0]
        group_selectivities_t[t_idx] = get_group_selectivity(W_t[t_idx],net.group_id)#-get_group_selectivity(W_t[t_idx*10],control_group)

#    plt.plot(selectivities_t[::10,net.pop_coupling<2.0],color='b',alpha=0.5)
#    plt.plot(selectivities_t[::10,net.pop_coupling>6.0],color='r',alpha=0.5)
#    plt.plot(np.mean(selectivities_t[::10,net.pop_coupling<2.0],axis=1),'b',lw=4)
#    plt.plot(np.mean(selectivities_t[::10,net.pop_coupling>6.0],axis=1),'r',lw=4)
#
#    plt.show()
#
#    plt.plot(group_selectivities_t[::10,net.pop_coupling<2.0],color='b',alpha=0.5)
#    plt.plot(group_selectivities_t[::10,net.pop_coupling>6.0],color='r',alpha=0.5)
#    plt.plot(np.mean(group_selectivities_t[::10,net.pop_coupling<2.0],axis=1),'b',lw=4)
#    plt.plot(np.mean(group_selectivities_t[::10,net.pop_coupling>6.0],axis=1),'r',lw=4)
#    plt.show()


#    stim_response = np.zeros(int(T/stim_period))
#
#    for stim_idx in xrange(int(T/stim_period)):
#        stim_response[stim_idx] = (np.mean(x_t[stim_idx*(stim_period):(stim_idx+1)*stim_period,10*presented_bar[stim_idx]:10*(presented_bar[stim_idx]+1)]))
#    plt.scatter(np.arange(len(presented_bar))[presented_bar==5],stim_response[presented_bar==5],color='r')
#    plt.plot(np.arange(len(presented_bar))[presented_bar!=5],stim_response[presented_bar!=5],color='b')
#    plt.plot(np.arange(len(presented_bar)),stim_response,'b--')
#
#    plt.scatter(np.arange(len(presented_paired))[presented_paired==5],stim_response[presented_paired==5],color='g')
#
#    plt.show()
    results = {
        'selectivities_t': selectivities_t,
        'group_selectivities_t': group_selectivities_t
    }

    return results

def analyse_RF_net_run_ori_group(run_results,net,N_ori=10):
    #W_t = run_results['W_t']
    #W_plastic = run_results['W_plastic']
    x_t = run_results['x_t']
    T = run_results['T']
    t_res = run_results['t_res']

    results = {}

#    plt.plot(W_t[:,:20,3],'g')
#    plt.plot(W_t[:,:20,2],'r')
#    plt.plot(W_t[:,:20,9],'b')
#    plt.show()

#    selectivities_t = np.zeros((int(T/t_res),net.N_filters))
#    for t_idx in xrange(int(T/t_res)):
#        selectivities_t[t_idx] = get_selectivity(np.transpose(W_t[t_idx]),N_ori,0)[0]

#    plt.plot(selectivities_t[::10,net.pop_coupling<2.0],color='b',alpha=0.5)
#    plt.plot(selectivities_t[::10,net.pop_coupling>6.0],color='r',alpha=0.5)
#    plt.plot(np.mean(selectivities_t[::10,net.pop_coupling<2.0],axis=1),'b',lw=4)
#    plt.plot(np.mean(selectivities_t[::10,net.pop_coupling>6.0],axis=1),'r',lw=4)
#    plt.show()
#
#    stim_response = np.zeros(int(T/stim_period))
#
#    for stim_idx in xrange(int(T/stim_period)):
#        stim_response[stim_idx] = (np.mean(x_t[stim_idx*(stim_period):(stim_idx+1)*stim_period,10*presented_bar[stim_idx]:10*(presented_bar[stim_idx]+1)]))
#    plt.scatter(np.arange(len(presented_bar))[presented_bar==5],stim_response[presented_bar==5],color='r')
#    plt.plot(np.arange(len(presented_bar))[presented_bar!=5],stim_response[presented_bar!=5],color='b')
#    plt.plot(np.arange(len(presented_bar)),stim_response,'b--')
#
#    plt.scatter(np.arange(len(presented_paired))[presented_paired==5],stim_response[presented_paired==5],color='g')
#

    pop_rate = np.mean(x_t,axis=1)
    empirical_pop_coupling_excl = np.zeros(net.N_filters)

    for i in xrange(net.N_filters):
        pop_rate_excl = np.mean(np.array([x_t[:,idx] for idx in range(x_t.shape[1]) if idx not in [i]]),axis=0)
        empirical_pop_coupling_excl[i] = scipy.stats.pearsonr(pop_rate_excl,x_t[:,i])[0]

    results['empirical_pop_coupling'] = empirical_pop_coupling_excl
    results['pop_coupling_rval'],results['pop_coupling_pval'] = scipy.stats.pearsonr(net.pop_coupling,empirical_pop_coupling_excl)
    results['rate_rval'],results['rate_pval'] = scipy.stats.pearsonr(net.pop_coupling,np.mean(x_t,axis=0))

#    plt.scatter(pop_coupling,stats.zscore(empirical_pop_coupling))
#    plt.title(str(scipy.stats.pearsonr(pop_coupling,empiral_pop_coupling)[0])+', pval = ' + str(scipy.stats.pearsonr(pop_coupling,empiral_pop_coupling)[1]))
#    plt.show()


#    plt.scatter(pop_coupling,np.mean(x_t,axis=0))
#    plt.title(str(scipy.stats.pearsonr(pop_coupling,np.mean(x_t,axis=0))[0])+', pval = ' + str(scipy.stats.pearsonr(pop_coupling,np.mean(x_t,axis=0))[1]))
#    plt.show()

    results['pop_coupling_partial_rval'], results['pop_coupling_partial_pval'] = partial_corr(net.pop_coupling,empirical_pop_coupling_excl,np.mean(x_t,axis=0),False)


#    results = {
#        'selectivities_t': selectivities_t,
#        'pop_coupling_rval': ,
#        'pop_coupling_pval': ,
#        'rate_rval': ,
#        'rate_pval': ,
#        'pop_coupling_partial_rval':
#        'pop_coupling_partial_pval':
#    }

    return results

def run_exp_group_selectivity(sim_pars,N_sims=10):
    selectivities_results_corr = []
    selectivities_results_uncorr = []
    selectivities_results_anticorr = []
    networks = []

    fig,axes = plt.subplots(3,3)
    sea_colors = sns.color_palette(n_colors=N_sims)

    for idx in xrange(N_sims):
        print 'running sim idx ', idx
        net_i = generate_RF_net(sim_pars)
        networks.append(net_i)

        simresults_corr_i = run_RF_net(net_i,'corr',T,group_corr_strength,t_res)
        simresults_uncorr_i = run_RF_net(net_i,'uncorr',T,group_corr_strength,t_res)
        simresults_anticorr_i = run_RF_net(net_i,'anticorr',T,group_corr_strength,t_res)

        selectivities_results_corr.append(plot_RF_net_run(simresults_corr_i,net_i))
        selectivities_results_uncorr.append(plot_RF_net_run(simresults_uncorr_i,net_i))
        selectivities_results_anticorr.append(plot_RF_net_run(simresults_anticorr_i,net_i))

#        axes[0].plot(selectivities_results_corr[-1]['group_selectivities_t'][::10,net_i.pop_coupling<2.0],'--',color=sea_colors[idx],alpha=0.2)
#        axes[1].plot(selectivities_results_uncorr[-1]['group_selectivities_t'][::10,net_i.pop_coupling<2.0],'--',color=sea_colors[idx],alpha=0.2)
#        axes[2].plot(selectivities_results_anticorr[-1]['group_selectivities_t'][::10,net_i.pop_coupling<2.0],'--',color=sea_colors[idx],alpha=0.2)
#        axes[0].plot(selectivities_results_corr[-1]['group_selectivities_t'][::10,net_i.pop_coupling>6.0],color=sea_colors[idx],alpha=0.2)
#        axes[1].plot(selectivities_results_uncorr[-1]['group_selectivities_t'][::10,net_i.pop_coupling>6.0],color=sea_colors[idx],alpha=0.2)
#        axes[2].plot(selectivities_results_anticorr[-1]['group_selectivities_t'][::10,net_i.pop_coupling>6.0],color=sea_colors[idx],alpha=0.2)
#
        axes[0][0].plot(np.mean(selectivities_results_corr[-1]['group_selectivities_t'][:,net_i.pop_coupling<np.mean(net_i.pop_coupling)],axis=1),'--',color=sea_colors[idx],lw=4)
        axes[0][1].plot(np.mean(selectivities_results_uncorr[-1]['group_selectivities_t'][:,net_i.pop_coupling<np.mean(net_i.pop_coupling)],axis=1),'--',color=sea_colors[idx],lw=4)
        axes[0][2].plot(np.mean(selectivities_results_anticorr[-1]['group_selectivities_t'][:,net_i.pop_coupling<np.mean(net_i.pop_coupling)],axis=1),'--',color=sea_colors[idx],lw=4)
        axes[0][0].plot(np.mean(selectivities_results_corr[-1]['group_selectivities_t'][:,net_i.pop_coupling>np.mean(net_i.pop_coupling)],axis=1),color=sea_colors[idx],lw=4)
        axes[0][1].plot(np.mean(selectivities_results_uncorr[-1]['group_selectivities_t'][:,net_i.pop_coupling>np.mean(net_i.pop_coupling)],axis=1),color=sea_colors[idx],lw=4)
        axes[0][2].plot(np.mean(selectivities_results_anticorr[-1]['group_selectivities_t'][:,net_i.pop_coupling>np.mean(net_i.pop_coupling)],axis=1),color=sea_colors[idx],lw=4)

        axes[0][0].set_title('Group selectivity, corr')
        axes[0][1].set_title('Group selectivity, uncorr')
        axes[0][2].set_title('Group selectivity, anticorr')

        axes[1][0].plot(np.mean(selectivities_results_corr[-1]['group_selectivities_t'][:,:],axis=1),color=sea_colors[idx],lw=4)
        axes[1][0].plot(np.mean(selectivities_results_anticorr[-1]['group_selectivities_t'][:,:],axis=1),'--',color=sea_colors[idx],lw=4)
        axes[1][0].plot(np.mean(selectivities_results_uncorr[-1]['group_selectivities_t'][:,:],axis=1),color=sea_colors[idx],lw=4,alpha=0.3)
        axes[1][0].set_title('Group selectivity entire net')

        axes[1][1].plot(np.mean(selectivities_results_corr[-1]['selectivities_t'][:,:],axis=1),color=sea_colors[idx],lw=4)
        axes[1][1].plot(np.mean(selectivities_results_anticorr[-1]['selectivities_t'][:,:],axis=1),'--',color=sea_colors[idx],lw=4)
        axes[1][1].plot(np.mean(selectivities_results_uncorr[-1]['selectivities_t'][:,:],axis=1),color=sea_colors[idx],lw=4,alpha=0.3)
        axes[1][1].set_title('Selectivity entire net')

        axes[1][2].hist(selectivities_results_corr[-1]['group_selectivities_t'][-1,:],10,color='b',histtype='step',range=(-1.0,1.0),lw=4)
        axes[1][2].hist(selectivities_results_uncorr[-1]['group_selectivities_t'][-1,:],10,color='g',histtype='step',range=(-1.0,1.0),lw=4)
        axes[1][2].hist(selectivities_results_anticorr[-1]['group_selectivities_t'][-1,:],10,color='r',histtype='step',range=(-1.0,1.0),lw=4)

    #plt.show()

    exp_results = {
        'networks': networks,
        'selectivities_results_corr': selectivities_results_corr,
        'selectivities_results_uncorr': selectivities_results_uncorr,
        'selectivities_results_anticorr': selectivities_results_anticorr
    }

    return exp_results

def plot_exp_group_selectivity(exp_results,N_sims=10,T=2000,N_filters=200,group_corr_size=50,group_corr_strength=2.5):
    fig,axes = plt.subplots(3,3)
    sea_colors = sns.color_palette(n_colors=N_sims)

    networks = exp_results['networks']
    selectivities_results_corr = exp_results['selectivities_results_corr']
    selectivities_results_uncorr = exp_results['selectivities_results_uncorr']
    selectivities_results_anticorr = exp_results['selectivities_results_anticorr']

    for idx in xrange(N_sims):
        net_i = networks[idx][0]
#        axes[0].plot(selectivities_results_corr[-1]['group_selectivities_t'][::10,net_i.pop_coupling<2.0],'--',color=sea_colors[idx],alpha=0.2)
#        axes[1].plot(selectivities_results_uncorr[-1]['group_selectivities_t'][::10,net_i.pop_coupling<2.0],'--',color=sea_colors[idx],alpha=0.2)
#        axes[2].plot(selectivities_results_anticorr[-1]['group_selectivities_t'][::10,net_i.pop_coupling<2.0],'--',color=sea_colors[idx],alpha=0.2)
#        axes[0].plot(selectivities_results_corr[-1]['group_selectivities_t'][::10,net_i.pop_coupling>6.0],color=sea_colors[idx],alpha=0.2)
#        axes[1].plot(selectivities_results_uncorr[-1]['group_selectivities_t'][::10,net_i.pop_coupling>6.0],color=sea_colors[idx],alpha=0.2)
#        axes[2].plot(selectivities_results_anticorr[-1]['group_selectivities_t'][::10,net_i.pop_coupling>6.0],color=sea_colors[idx],alpha=0.2)
#
        axes[0][0].plot(np.mean(selectivities_results_corr[idx][0]['group_selectivities_t'][:,net_i.pop_coupling<np.mean(net_i.pop_coupling)],axis=1),'--',color=sea_colors[idx],lw=4)
        axes[0][1].plot(np.mean(selectivities_results_uncorr[idx][0]['group_selectivities_t'][:,net_i.pop_coupling<np.mean(net_i.pop_coupling)],axis=1),'--',color=sea_colors[idx],lw=4)
        axes[0][2].plot(np.mean(selectivities_results_anticorr[idx][0]['group_selectivities_t'][:,net_i.pop_coupling<np.mean(net_i.pop_coupling)],axis=1),'--',color=sea_colors[idx],lw=4)
        axes[0][0].plot(np.mean(selectivities_results_corr[idx][0]['group_selectivities_t'][:,net_i.pop_coupling>np.mean(net_i.pop_coupling)],axis=1),color=sea_colors[idx],lw=4)
        axes[0][1].plot(np.mean(selectivities_results_uncorr[idx][0]['group_selectivities_t'][:,net_i.pop_coupling>np.mean(net_i.pop_coupling)],axis=1),color=sea_colors[idx],lw=4)
        axes[0][2].plot(np.mean(selectivities_results_anticorr[idx][0]['group_selectivities_t'][:,net_i.pop_coupling>np.mean(net_i.pop_coupling)],axis=1),color=sea_colors[idx],lw=4)

        axes[0][0].set_title('Group selectivity, corr')
        axes[0][1].set_title('Group selectivity, uncorr')
        axes[0][2].set_title('Group selectivity, anticorr')

        axes[1][0].plot(np.mean(selectivities_results_corr[idx][0]['group_selectivities_t'][:,:],axis=1),color=sea_colors[idx],lw=4)
        axes[1][0].plot(np.mean(selectivities_results_anticorr[idx][0]['group_selectivities_t'][:,:],axis=1),'--',color=sea_colors[idx],lw=4)
        axes[1][0].plot(np.mean(selectivities_results_uncorr[idx][0]['group_selectivities_t'][:,:],axis=1),color=sea_colors[idx],lw=4,alpha=0.3)
        axes[1][0].set_title('Group selectivity entire net')

        axes[1][1].plot(np.mean(selectivities_results_corr[idx][0]['selectivities_t'][:,:],axis=1),color=sea_colors[idx],lw=4)
        axes[1][1].plot(np.mean(selectivities_results_anticorr[idx][0]['selectivities_t'][:,:],axis=1),'--',color=sea_colors[idx],lw=4)
        axes[1][1].plot(np.mean(selectivities_results_uncorr[idx][0]['selectivities_t'][:,:],axis=1),color=sea_colors[idx],lw=4,alpha=0.3)
        axes[1][1].set_title('Selectivity entire net')

        axes[1][2].hist(selectivities_results_corr[idx][0]['group_selectivities_t'][-1,:],10,color='b',histtype='step',range=(-1.0,1.0),lw=4)
        axes[1][2].hist(selectivities_results_uncorr[idx][0]['group_selectivities_t'][-1,:],10,color='g',histtype='step',range=(-1.0,1.0),lw=4)
        axes[1][2].hist(selectivities_results_anticorr[idx][0]['group_selectivities_t'][-1,:],10,color='r',histtype='step',range=(-1.0,1.0),lw=4)

    #plt.show()


def run_exp_pop_coupling(sim_pars,N_sims=3,T_pc_sampling=5000):
    selectivities_results_corr = []
    selectivities_results_uncorr = []
    selectivities_results_anticorr = []
    selectivities_results_uniform = []
    simresults_corr = []
    simresults_uniform = []
    selectivities_t_corr = []
    selectivities_t_uniform = []
    networks = []

    sim_pars_pc_measure = sim_pars.copy()
    sim_pars_pc_measure.update(sim_pars['pc_measure_pars'])

    for idx in xrange(N_sims):
        print 'running sim idx ', idx
        net_i = generate_RF_net(sim_pars,sim_pars['N_filters'],sim_pars['N_per_group'],sim_pars['pop_coupling_add_factor'],sim_pars['pop_coupling_mult_factor'],50)
        networks.append(net_i)

        simresults_corr_i = run_RF_net(net_i,sim_pars,'corr',sim_pars['T'],sim_pars['sample_res'])
        simresults_corr.append(simresults_corr_i)
        if not sim_pars['prune_weights'] and not sim_pars['only_diverse'] :
            simresults_uncorr_i = run_RF_net(net_i,sim_pars,'uncorr',sim_pars['T'],sim_pars['sample_res'])
            simresults_anticorr_i = run_RF_net(net_i,sim_pars,'anticorr',sim_pars['T'],sim_pars['sample_res'])
        simresults_uniform_i = run_RF_net(net_i,sim_pars,'uniform',sim_pars['T'],sim_pars['sample_res'])

        net_i.W = simresults_corr_i['W_plastic']
        net_i.W_IE = simresults_corr_i['W_IE_t'][-1].reshape((net_i.N_filters,1))
        simresults_corr_ix = run_RF_net(net_i,sim_pars_pc_measure,'corr',T_pc_sampling,1,True)

        if not sim_pars['prune_weights'] and not sim_pars['only_diverse']:
            net_i.W = simresults_uncorr_i['W_plastic']
            net_i.W_IE = simresults_uncorr_i['W_IE_t'][-1].reshape((net_i.N_filters,1))
            simresults_uncorr_ix = run_RF_net(net_i,sim_pars_pc_measure,'uncorr',T_pc_sampling,1,True)

            net_i.W = simresults_anticorr_i['W_plastic']
            net_i.W_IE = simresults_anticorr_i['W_IE_t'][-1].reshape((net_i.N_filters,1))
            simresults_anticorr_ix = run_RF_net(net_i,sim_pars_pc_measure,'anticorr',T_pc_sampling,1,True)

        net_i.W = simresults_uniform_i['W_plastic']
        net_i.W_IE = simresults_uniform_i['W_IE_t'][-1].reshape((net_i.N_filters,1))
        simresults_uniform_ix = run_RF_net(net_i,sim_pars_pc_measure,'uniform',T_pc_sampling,1,True)
        simresults_uniform.append(simresults_uniform_i)

        selectivities_results_corr.append(analyse_RF_net_run_ori_group(simresults_corr_ix,net_i))
        if not sim_pars['prune_weights'] and not sim_pars['only_diverse'] :
            selectivities_results_uncorr.append(analyse_RF_net_run_ori_group(simresults_uncorr_ix,net_i))
            selectivities_results_anticorr.append(analyse_RF_net_run_ori_group(simresults_anticorr_ix,net_i))
        selectivities_results_uniform.append(analyse_RF_net_run_ori_group(simresults_uniform_ix,net_i))

        temp_selectivities_t_corr = np.zeros((int(sim_pars['T']/sim_pars['sample_res']),net_i.N_filters))
        if not sim_pars['prune_weights'] and not sim_pars['only_diverse']:
            selectivities_t_anticorr = np.zeros((int(sim_pars['T']/sim_pars['sample_res']),net_i.N_filters))
            selectivities_t_uncorr = np.zeros((int(sim_pars['T']/sim_pars['sample_res']),net_i.N_filters))
        temp_selectivities_t_uniform = np.zeros((int(sim_pars['T']/sim_pars['sample_res']),net_i.N_filters))
        for t_idx in xrange(int(sim_pars['T']/sim_pars['sample_res'])):
            temp_selectivities_t_corr[t_idx] = get_selectivity(np.transpose(simresults_corr_i['W_t'][t_idx]),int(sim_pars['N_filters']/sim_pars['N_per_group']),0)[0]
            if not sim_pars['prune_weights'] and not sim_pars['only_diverse']:
                selectivities_t_uncorr[t_idx] = get_selectivity(np.transpose(simresults_uncorr_i['W_t'][t_idx]),int(sim_pars['N_filters']/sim_pars['N_per_group']),0)[0]
                selectivities_t_anticorr[t_idx] = get_selectivity(np.transpose(simresults_anticorr_i['W_t'][t_idx]),int(sim_pars['N_filters']/sim_pars['N_per_group']),0)[0]
            temp_selectivities_t_uniform[t_idx] = get_selectivity(np.transpose(simresults_uniform_i['W_t'][t_idx]),int(sim_pars['N_filters']/sim_pars['N_per_group']),0)[0]
        selectivities_t_corr.append(temp_selectivities_t_corr)
        selectivities_t_uniform.append(temp_selectivities_t_uniform)

    exp_results = {
        'networks': networks,
        'selectivities_results_corr': selectivities_results_corr,
        #'selectivities_results_uncorr': selectivities_results_uncorr,
        #'selectivities_results_anticorr': selectivities_results_anticorr,
        'selectivities_results_uniform': selectivities_results_uniform,
        'selectivities_t_corr': selectivities_t_corr,
        #'selectivities_t_uncorr': selectivities_t_uncorr,
        #'selectivities_t_anticorr': selectivities_t_anticorr,
        'selectivities_t_uniform': selectivities_t_uniform,
       # 'simresults_anticorr': simresults_anticorr_i,
        'simresults_corr': simresults_corr,
       # 'simresults_uncorr': simresults_uncorr_i,
        'simresults_uniform': simresults_uniform,
       # 'W_anticorr': simresults_anticorr_i['W_plastic'],
        'W_corr': simresults_corr_i['W_plastic'],
       # 'W_uncorr': simresults_uncorr_i['W_plastic'],
        'W_uniform': simresults_uniform_i['W_plastic']

    }

    if not sim_pars['prune_weights'] and not sim_pars['only_diverse']:
        exp_results['selectivities_results_uncorr'] = selectivities_results_uncorr
        exp_results['selectivities_results_anticorr'] = selectivities_results_anticorr
        exp_results['selectivities_t_uncorr'] = selectivities_t_uncorr
        exp_results['selectivities_t_anticorr'] = selectivities_t_anticorr
        exp_results['simresults_uncorr'] = simresults_uncorr_i
        exp_results['simresults_anticorr'] = simresults_anticorr_i
        exp_results['W_uncorr'] =  simresults_uncorr_i['W_plastic']
        exp_resluts['W_anticorr'] =  simresults_anticorr_i['W_plastic']

    return exp_results




