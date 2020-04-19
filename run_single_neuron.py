from NeuroTools import stgen
import numpy as np
from matplotlib import pyplot as plt

from scipy import stats

import seaborn as sns

r_0 = 1.0
r_max = 20

dt = 1.0
stim_time = 100

def get_rates(_x):
    _x[_x<=0] = r_0*np.tanh(_x[_x<=0]/r_0)
    _x[_x>0] = (r_max-r_0)*np.tanh(_x[_x>0]/(r_max-r_0))

    return _x

def get_rate(_x):
    if _x < 0:
        return r_0*np.tanh(_x/r_0)
    else:
        return (r_max-r_0)*np.tanh(_x/(r_max-r_0))

    return _x

def run_sim_clean(simpars_pass):
    pars = {}

    pars.update(simpars_pass)

    sample_res = pars['sample_res']

    N_recurrent_neurons = pars['N_recurrent_neurons']

    #w_IE_single = pars['w_IE_single']
    w_IE_single = pars['w_IE_single']

    phi_FF = np.pi

    phi_reward = 0
    phi_target = pars['phi_rec']

    x_rec = np.zeros((1,N_recurrent_neurons))
    H_FF_rec = np.zeros((1,N_recurrent_neurons))
    H_TD_reward = 0.0
    W_rec = np.ones((N_recurrent_neurons,1))*pars['W_rec_static']
    W_rec_static = W_rec.copy()
    W_rec_slow_single = W_rec.copy()
    w_IE_rec = np.ones((1,N_recurrent_neurons))*pars['w_IE']
    w_IE_target = np.ones((1,N_recurrent_neurons))*pars['w_IE']
    W_target = np.ones((N_recurrent_neurons,1))*0.5*pars['W_rec_max']
    w_EI= pars['w_EI']
    w_TD_down = pars['w_TD_down']
    w_TD_up = pars['w_TD_up']

    x_single = 0.0
    x_TD = 0.0
    x_inh = 0.0
    x_target = np.zeros((1,N_recurrent_neurons))

    theta_BCM_single = pars['theta_BCM_single']

    theta_BCM_rec = np.ones((1,N_recurrent_neurons))*pars['BCM_target']
    W_single_rec = np.ones((1,N_recurrent_neurons))*1.0/N_recurrent_neurons

    sig = stgen.StGen()
    OU_noise = sig.OU_generator_weave1(1,10.0,pars['sigma_OU'],0,0,pars['T'])[0]
    OU_noise = np.transpose(OU_noise)

    ind_OU_noise_single = sig.OU_generator_weave1(1,100.0,pars['ind_sigma_OU'],0,0,pars['T'])[0]
    ind_OU_noise_single = np.transpose(ind_OU_noise_single)
    ind_OU_noise_rec = np.zeros((pars['T'],N_recurrent_neurons))
    for i in xrange(N_recurrent_neurons):
        ind_OU_noise_rec[:,i] = sig.OU_generator_weave1(1,100.0,pars['ind_sigma_OU'],0,0,pars['T'])[0]
    ind_OU_noise_target = np.zeros((pars['T'],N_recurrent_neurons))
    for i in xrange(N_recurrent_neurons):
        ind_OU_noise_target[:,i] = sig.OU_generator_weave1(1,100.0,pars['ind_sigma_OU'],0,0,pars['T'])[0]

    rates_single = np.zeros(int(pars['T']/sample_res))
    rates_inh = np.zeros(int(pars['T']/sample_res))
    rates_TD = np.zeros(int(pars['T']/sample_res))
    rates_rec = np.zeros((int(pars['T']/sample_res),N_recurrent_neurons))
    rates_target = np.zeros((int(pars['T']/sample_res),N_recurrent_neurons))
    weights_rec = np.zeros((int(pars['T']/sample_res),N_recurrent_neurons))
    theta_BCM_singles_samples = np.zeros(int(pars['T']/sample_res))
    weights_inh_single = np.zeros(int(pars['T']/sample_res))
    weights_inh_rec = np.zeros((int(pars['T']/sample_res),N_recurrent_neurons))
    weights_inh_target = np.zeros((int(pars['T']/sample_res),N_recurrent_neurons))
    weights_target = np.zeros((int(pars['T']/sample_res),N_recurrent_neurons))

    for t_idx in xrange(pars['T']):

        if t_idx%stim_time == 0:
            stim_orientation = pars['phi_rec'][np.random.randint(len(pars['phi_rec']))]
            if t_idx < pars['T_reward_start']:
                H_FF_single = (np.cos(stim_orientation-phi_FF)+1.0)*pars['H_FF_stimulus']/2
                H_FF_rec = (np.cos(stim_orientation-pars['phi_rec'])+1.0)*pars['H_FF_stimulus_rec']/2
                H_FF_target = (np.cos(stim_orientation-phi_target)+1.0)*pars['H_FF_stimulus_rec']/2
            #if t_idx > pars['T_reward_start']:
            else:
                if np.random.rand() > 0.5:
                    stim_orientation = phi_reward
                H_TD_reward = (stim_orientation == phi_reward)*pars['H_TD_reward_stimulus']
                H_TD_reward = (np.cos(stim_orientation-phi_reward))*pars['H_TD_reward_stimulus']
                H_FF_single = (np.cos(stim_orientation-phi_FF)+1.0)*pars['H_FF_stimulus']/2
                H_FF_rec = (np.cos(stim_orientation-pars['phi_rec'])+1.0)*pars['H_FF_stimulus_rec']/2
                H_FF_target = (np.cos(stim_orientation-phi_target)+1.0)*pars['H_FF_stimulus_rec']/2

        x_single += dt*(-1*x_single + get_rate(H_FF_single + w_TD_down*x_TD + pars['c']*np.dot(x_rec,W_rec_static) + (w_IE_single*x_inh) + OU_noise[t_idx] + ind_OU_noise_single[t_idx] + 0.0*np.dot(x_rec,W_target))) # only scaling inhibition by coupling constant
        #x_single += dt*(-1*x_single + get_rate(H_FF_single + w_TD_down*x_TD + pars['c']*np.dot(x_rec,W_rec_slow_single) + (w_IE_single*x_inh) + OU_noise[t_idx] + ind_OU_noise_single[t_idx] + 0.0*np.dot(x_rec,W_target))) # only scaling inhibition by coupling constant

        #x_TD += dt*(-1*x_TD + get_rate(H_TD_reward + w_TD_up*x_single + w_TD_up*np.sum(x_rec)))
        x_TD = H_TD_reward

        x_rec += dt*(-1*x_rec + get_rates(H_FF_rec + w_TD_down*x_TD + np.dot(x_inh,w_IE_rec) + pars['c_rec']*(np.dot(x_rec,W_rec_static)) + OU_noise[t_idx] + ind_OU_noise_rec[t_idx])) #So far, no inputs from single neuron. So it is independent of it (given the top-down stimulus)

        x_inh += dt*(-1*x_inh + get_rate(w_EI*x_single + w_EI*np.sum(x_rec) + w_EI*np.sum(x_target)))# + OU_noise[t_idx])) # global inhibition (?)

        x_target += dt*(-1*x_target + get_rates(H_FF_target + w_TD_down*x_TD + pars['c_target']*np.dot(x_rec,W_rec_static) + (np.dot(x_inh,w_IE_target)) + OU_noise[t_idx] + ind_OU_noise_target[t_idx])) # only scaling inhibition by coupling constant

        x_rec[x_rec<0] = 0
        x_TD = max(0,x_TD)
        x_single= max(0,x_single)
        x_inh = max(0,x_inh)
        x_target[x_target<0] = 0

        #W_rec += (1.0/c)*alpha*np.transpose(x_rec)*x_single*(x_single-theta_BCM_single)

        if t_idx > pars['T_exc_plasticity_start']:
            #W_rec[pars['plastic_weights_indices']] += pars['alpha']*np.transpose(x_rec)[pars['plastic_weights_indices']]*x_single*(x_single-theta_BCM_single)
            #W_rec[pars['plastic_weights_indices']] = pars['decay_lambda']*W_rec[pars['plastic_weights_indices']]
            #W_rec[W_rec<0] = 0
            #W_rec[W_rec> pars['W_rec_max']] = pars['W_rec_max']
            #W_target += pars['alpha']*np.transpose(x_target)*x_single*(x_single-theta_BCM_single)
            W_target += pars['alpha']*np.transpose(x_target)*x_single
            W_target = pars['decay_lambda']*W_target
            W_target[W_target<0] = 0 #min(w_target,pars['W_rec_max'])
            W_target[W_target> pars['W_rec_max']] = pars['W_rec_max']

            W_target *= 2.0*pars['W_rec_max']/sum(W_target)

            if pars['synaptic_scaling']:
                #if ['postsyn_scaling']:
                postsyn_tot = np.sum(W_rec_plastic,axis=1).reshape(N_recurrent_neurons,1)
                #W_rec_plastic += pars['scaling_rate']*(pars['scaling_target'] - postsyn_tot)
                W_target += pars['scaling_rates_i']*(pars['scaling_target'] - postsyn_tot)
                if pars['outgoing_scaling']:
                    presyn_tot = np.sum(W_rec_plastic,axis=0).reshape(1,N_recurrent_neurons)
                    W_target += pars['scaling_rates_i']*(pars['scaling_target'] - presyn_tot)

            #theta_BCM_single += theta_BCM_dt*((x_single/BCM_target)*x_single - theta_BCM_single)

        if t_idx < pars['T_reward_start'] or True:
            w_IE_single += -pars['eta']*(x_inh*(x_single-pars['BCM_target']))
            w_IE_single[w_IE_single< pars['W_inh_min']] = pars['W_inh_min']
            w_IE_single[w_IE_single>0] = 0.0

        w_IE_rec += -pars['eta']*(x_inh*(x_rec-pars['BCM_target']))
        w_IE_rec[w_IE_rec<pars['W_inh_min']] = pars['W_inh_min']
        # Letting inh weights be positive too, for full rate control
        #w_IE_rec[w_IE_rec>0] = 0.0

        w_IE_target += -pars['eta']*(x_inh*(x_target-pars['BCM_target']))
        w_IE_target[w_IE_target<pars['W_inh_min']] = pars['W_inh_min']
        w_IE_target[w_IE_target>0] = 0.0

        if t_idx%sample_res == 0:
            rates_single[int(t_idx/sample_res)] = x_single
            rates_rec[int(t_idx/sample_res)] = x_rec.reshape(N_recurrent_neurons)
            rates_target[int(t_idx/sample_res)] = x_target.reshape(N_recurrent_neurons)
            rates_TD[int(t_idx/sample_res)] = x_TD
            rates_inh[int(t_idx/sample_res)] = x_inh
            #weights_rec[int(t_idx/sample_res)] = W_rec.reshape(N_recurrent_neurons)
            weights_target[int(t_idx/sample_res)] = W_target.reshape(N_recurrent_neurons)
            theta_BCM_singles_samples[int(t_idx/sample_res)] = theta_BCM_single
            weights_inh_single[int(t_idx/sample_res)] = w_IE_single
            weights_inh_rec[int(t_idx/sample_res)] = w_IE_rec.reshape(N_recurrent_neurons)
            weights_inh_target[int(t_idx/sample_res)] = w_IE_target.reshape(N_recurrent_neurons)

    results = {
        #'weights_rec': weights_rec,
        'weights_target': weights_target,
        'rates_single': rates_single,
        'rates_rec': rates_rec,
        'rates_TD': rates_TD,
        'rates_target': rates_target,
        'rates_inh': rates_inh,
        'weights_inh_single': weights_inh_single,
        'weights_inh_rec': weights_inh_rec,
        'weights_inh_target': weights_inh_target,
        'theta_BCM_singles': theta_BCM_singles_samples,
        'sim_pars': pars
    }

    return results

def run_sim_diverse_learning_rates(simpars_pass):
    pars = {}

    pars.update(simpars_pass)

    sample_res = pars['sample_res']

    N_recurrent_neurons = pars['N_recurrent_neurons']

    w_IE_single = pars['w_IE_single']

    phi_FF = np.pi

    phi_reward = 0
    phi_target = pars['phi_rec']
    phi_reward_idx = 0

    x_rec = np.zeros((1,N_recurrent_neurons))
    H_FF_rec = np.zeros((1,N_recurrent_neurons))
    H_TD_reward = 0.0
    W_rec = np.ones((N_recurrent_neurons,1))*pars['W_rec_static']
    W_rec_plastic = np.ones((N_recurrent_neurons,N_recurrent_neurons))*pars['W_rec_static']
    np.fill_diagonal(W_rec_plastic,0.0)
    w_IE_rec = np.ones((1,N_recurrent_neurons))*pars['w_IE']
    w_IE_target = np.ones((1,N_recurrent_neurons))*pars['w_IE']
    W_target = np.ones((N_recurrent_neurons,1))*0.5*pars['W_rec_max']
    w_EI= pars['w_EI']
    w_TD_down = pars['w_TD_down']
    w_TD_up = pars['w_TD_up']

    x_single = 0.0
    x_TD = 0.0
    x_inh = 0.0
    x_target = np.zeros((1,N_recurrent_neurons))

    theta_BCM_single = pars['theta_BCM_single']

    theta_BCM_rec = np.ones((1,N_recurrent_neurons))*pars['BCM_target']
    W_single_rec = np.ones((1,N_recurrent_neurons))*1.0/N_recurrent_neurons

    sig = stgen.StGen()
    OU_noise = sig.OU_generator_weave1(1,10.0,pars['sigma_OU'],0,0,pars['T'])[0]
    OU_noise = np.transpose(OU_noise)

    ind_OU_noise_single = sig.OU_generator_weave1(1,100.0,pars['ind_sigma_OU'],0,0,pars['T'])[0]
    ind_OU_noise_single = np.transpose(ind_OU_noise_single)
    ind_OU_noise_rec = np.zeros((pars['T'],N_recurrent_neurons))
    for i in xrange(N_recurrent_neurons):
        ind_OU_noise_rec[:,i] = sig.OU_generator_weave1(1,pars['ind_tau_OU'],pars['ind_sigma_OU'],0,0,pars['T'])[0]+ pars['ind_mean_OU']
    #ind_OU_noise_target = np.zeros((pars['T'],N_recurrent_neurons))
    #for i in xrange(N_recurrent_neurons):
    #    ind_OU_noise_target[:,i] = sig.OU_generator_weave1(1,100.0,pars['ind_sigma_OU'],0,0,pars['T'])[0]

    if pars['set_W_rec_plastic']:
        W_rec_plastic = pars['W_rec_plastic_passed'].copy()

    rates_single = np.zeros(int(pars['T']/sample_res))
    rates_inh = np.zeros(int(pars['T']/sample_res))
    rates_TD = np.zeros(int(pars['T']/sample_res))
    rates_rec = np.zeros((int(pars['T']/sample_res),N_recurrent_neurons))
    rates_target = np.zeros((int(pars['T']/sample_res),N_recurrent_neurons))
    weights_rec = np.zeros((int(pars['T']/sample_res),N_recurrent_neurons**2))
    #theta_BCM_singles_samples = np.zeros(int(pars['T']/sample_res))
    theta_BCM_rec_samples = np.zeros((int(pars['T']/sample_res),N_recurrent_neurons))
    weights_inh_single = np.zeros(int(pars['T']/sample_res))
    weights_inh_rec = np.zeros((int(pars['T']/sample_res),N_recurrent_neurons))
    weights_inh_target = np.zeros((int(pars['T']/sample_res),N_recurrent_neurons))
    weights_target = np.zeros((int(pars['T']/sample_res),N_recurrent_neurons))

    if pars['record_stim_responses']:
        presented_stims = np.zeros(int(pars['T']/stim_time))
        presented_rewards = np.zeros(int(pars['T']/stim_time))
        presented_stim_rec_rates = np.zeros((int(pars['T']/stim_time),N_recurrent_neurons))

    stim_orientation = -1

    for t_idx in xrange(pars['T']):
        if t_idx%pars['reward_switch_time'] == 0 and t_idx > pars['T_reward_start'] and t_idx < pars['T_reward_end']:
            phi_reward_idx = (phi_reward_idx+3)%len(pars['phi_stimuli'])
            phi_reward = pars['phi_stimuli'][phi_reward_idx]
            #phi_reward = pars['phi_rec'][np.random.randint(len(pars['phi_rec']))]
        if t_idx%stim_time == 0:
            if pars['record_stim_responses']:
                presented_stims[int(t_idx/stim_time)] = stim_orientation
                presented_stim_rec_rates[int(t_idx/stim_time)] = x_rec.reshape(N_recurrent_neurons)
                presented_rewards[int(t_idx/stim_time)] = phi_reward

            #stim_orientation = pars['phi_rec'][np.random.randint(len(pars['phi_rec']))]
            stim_orientation = pars['phi_stimuli'][np.random.randint(len(pars['phi_stimuli']))]
            #stim_orientation = np.random.uniform()*2*np.pi
            if pars['random_dynamic_input']:
                if t_idx%pars['dynamic_input_switch_time'] == 0:
                    H_FF_rec = np.random.uniform(0.0,pars['H_FF_stimulus_rec'],N_recurrent_neurons)
            elif t_idx < pars['T_reward_start'] or pars['reward_modulated_learning'] or t_idx > pars['T_reward_end']:
                H_TD_reward = 0
                H_FF_single = (np.cos(stim_orientation-phi_FF)+1.0)*pars['H_FF_stimulus']/2
                H_FF_target = (np.cos(stim_orientation-phi_target)+1.0)*pars['H_FF_stimulus_rec']/2
                #H_FF_rec = (np.cos(stim_orientation-pars['phi_rec'])+1.0)*pars['H_FF_stimulus_rec']/2
                #H_FF_rec = (stim_orientation == pars['phi_rec'])*pars['H_FF_stimulus_rec']
                #H_FF_rec = (stim_orientation == pars['phi_rec'])*pars['H_FF_stimulus_rec'] + pars['H_FF_baseline']
                # exponential, non-circular
                #H_FF_rec = np.exp(-2.0*(np.abs(stim_orientation-pars['phi_rec'])**2))*pars['H_FF_stimulus_rec']
                #circ_dist = min(abs(stim_orientation-pars['phi_rec']),abs(stim_orientation-pars['phi_rec']-np.pi),abs(pars['phi_rec']-stim_orientation-np.pi))
                #H_FF_rec = np.exp(-2.0*(circ_dist**2))*pars['H_FF_stimulus_rec']
                H_FF_rec = np.exp(-(1.0/(2*pars['tuning_width']**2))*(np.cos(stim_orientation-pars['phi_rec'])+1.0))*pars['H_FF_stimulus_rec']
                # von mises
                #H_FF_rec = np.exp(10.0*np.cos(stim_orientation-pars['phi_rec']))*pars['H_FF_stimulus_rec']/2
            #if t_idx > pars['T_reward_start']:
                #print stim_orientation,H_FF_rec
            else:
                #if np.random.rand() > 0.3:
                #    stim_orientation = phi_reward
                #H_TD_reward = (stim_orientation == phi_reward)*pars['H_Te_reward_stimulus']
                H_TD_reward = (np.cos(stim_orientation-phi_reward))*pars['H_TD_reward_stimulus']/2
                H_FF_single = (np.cos(stim_orientation-phi_FF)+1.0)*pars['H_FF_stimulus']/2
                H_FF_target = (np.cos(stim_orientation-phi_target)+1.0)*pars['H_FF_stimulus_rec']/2
                #H_FF_rec = (np.cos(stim_orientation-pars['phi_rec'])+1.0)*pars['H_FF_stimulus_rec']/2
                #H_FF_rec = (stim_orientation == pars['phi_rec'])*pars['H_FF_stimulus_rec']
                #H_FF_rec = (stim_orientation == pars['phi_rec'])*pars['H_FF_stimulus_rec'] + pars['H_FF_baseline']
                # exponential, non-circular
                #H_FF_rec = np.exp(-2.0*(np.abs(stim_orientation-pars['phi_rec'])**2))*pars['H_FF_stimulus_rec']
                #circ_dist = min(abs(stim_orientation-pars['phi_rec']),abs(stim_orientation-pars['phi_rec']-np.pi),abs(pars['phi_rec']-stim_orientation-np.pi))
                #H_FF_rec = np.exp(-2.0*(circ_dist**2))*pars['H_FF_stimulus_rec']
                H_FF_rec = np.exp(-(1.0/(2*pars['tuning_width']**2))*(np.cos(stim_orientation-pars['phi_rec'])+1.0))*pars['H_FF_stimulus_rec']


        x_rec += dt*(-1*x_rec + get_rates(H_FF_rec + w_TD_down*x_TD + np.dot(x_inh,w_IE_rec) + pars['c_rec']*(np.dot(W_rec_plastic,x_rec.transpose()).transpose()) + OU_noise[t_idx] + ind_OU_noise_rec[t_idx])) #So far, no inputs from single neuron. So it is independent of it (given the top-down stimulus)

        #x_single += dt*(-1*x_single + get_rate(H_FF_single + w_TD_down*x_TD + pars['c']*np.dot(x_rec,W_rec_static) + (w_IE_single*x_inh) + OU_noise[t_idx] + ind_OU_noise_single[t_idx] + 0.0*np.dot(x_rec,W_target))) # only scaling inhibition by coupling constant
        #x_single += dt*(-1*x_single + get_rate(H_FF_single + w_TD_down*x_TD + pars['c']*np.dot(x_rec,W_rec_slow_single) + (w_IE_single*x_inh) + OU_noise[t_idx] + ind_OU_noise_single[t_idx] + 0.0*np.dot(x_rec,W_target))) # only scaling inhibition by coupling constant

        x_TD += dt*(-1*x_TD + get_rate(H_TD_reward + w_TD_up*x_single + w_TD_up*np.sum(x_rec)))
        #x_TD = H_TD_reward

        #x_rec += dt*(-1*x_rec + get_rates(H_FF_rec + w_TD_down*x_TD + np.dot(x_inh,w_IE_rec) + pars['c_rec']*(np.dot(x_rec,W_rec_static)) + OU_noise[t_idx] + ind_OU_noise_rec[t_idx])) #So far, no inputs from single neuron. So it is independent of it (given the top-down stimulus)

        x_inh += dt*(-1*x_inh + get_rate(w_EI*x_single + w_EI*np.sum(x_rec) + w_EI*np.sum(x_target)))# + 0.5*w_EI*x_TD)# + OU_noise[t_idx])) # global inhibition (?)

        #x_target += dt*(-1*x_target + get_rates(H_FF_target + w_TD_down*x_TD + pars['c_target']*np.dot(x_rec,W_rec_static) + (np.dot(x_inh,w_IE_target)) + OU_noise[t_idx] + ind_OU_noise_target[t_idx])) # only scaling inhibition by coupling constant

        x_rec[x_rec<0] = 0
        x_TD = max(0,x_TD)
        #x_single= max(0,x_single)
        x_inh = max(0,x_inh)
        #x_target[x_target<0] = 0


        if t_idx > pars['T_exc_plasticity_start']:
            if pars['reward_modulated_learning'] and t_idx > pars['T_reward_start'] and t_idx < pars['T_reward_end']:
                #W_rec_plastic += pars['alpha_i']*x_rec.transpose()*x_rec*(stim_orientation == phi_reward)
                W_rec_plastic += pars['alpha_i']*x_rec.transpose()*x_rec*(x_rec-pars['theta_BCM_single'])*(stim_orientation == phi_reward)
            else:
                #W_rec_plastic += pars['alpha_i']*x_rec.transpose()*x_rec
                #W_rec_plastic += pars['alpha_i']*x_rec.transpose()*x_rec*(x_rec-theta_BCM_rec)
                W_rec_plastic += pars['alpha_i']*x_rec.transpose()*x_rec*(x_rec-pars['theta_BCM_single'])
                W_rec_plastic *= pars['decay_lambda']

            W_rec_plastic[W_rec_plastic<0] = 0.0

            # weight-dependent fluctuations
            #W_rec_plastic += stats.norm.rvs(scale=0.00001+W_rec_plastic*0.0005,size=(N_recurrent_neurons,N_recurrent_neurons))

            #theta_BCM_rec += pars['theta_BCM_dt']*((x_rec/pars['BCM_target'])*x_rec - theta_BCM_rec)
            W_rec_plastic[W_rec_plastic>pars['W_rec_max']]= pars['W_rec_max']
            #W_rec_plastic[W_rec_plastic>pars['W_max_scale'][W_rec_plastic]] = pars['W_max_scale']
            #W_rec_plastic[W_rec_plastic<0] = 0.0

            if pars['synaptic_scaling']:
                #if ['postsyn_scaling']:
                postsyn_tot = np.sum(W_rec_plastic,axis=1).reshape(N_recurrent_neurons,1)
                #W_rec_plastic += pars['scaling_rate']*(pars['scaling_target'] - postsyn_tot)
                W_rec_plastic += pars['scaling_rates_i']*(pars['scaling_target'] - postsyn_tot)
                if pars['outgoing_scaling']:
                    presyn_tot = np.sum(W_rec_plastic,axis=0).reshape(1,N_recurrent_neurons)
                    W_rec_plastic += pars['scaling_rates_i']*(pars['scaling_target'] - presyn_tot)

            #postsyn_tot = np.sum(W_rec_plastic,axis=1).reshape(N_recurrent_neurons,1)
            #postsyn_tot = np.multiply(postsyn_tot,1./pars['W_max_scale'].reshape(N_recurrent_neurons,1))
            #W_rec_plastic *= 2.0/postsyn_tot

            np.fill_diagonal(W_rec_plastic,0.0)

        w_IE_rec += -pars['eta']*(x_inh*(x_rec-pars['inh_target']))
        w_IE_rec[w_IE_rec<pars['W_inh_min']] = pars['W_inh_min']
        # Letting inh weights be positive too, for full rate control
        #w_IE_rec[w_IE_rec>0] = 0.0

        if t_idx%sample_res == 0:
            rates_single[int(t_idx/sample_res)] = x_single
            rates_rec[int(t_idx/sample_res)] = x_rec.reshape(N_recurrent_neurons)
            rates_target[int(t_idx/sample_res)] = x_target.reshape(N_recurrent_neurons)
            rates_TD[int(t_idx/sample_res)] = x_TD
            rates_inh[int(t_idx/sample_res)] = x_inh
            weights_rec[int(t_idx/sample_res)] = W_rec_plastic.reshape(N_recurrent_neurons**2)
            weights_target[int(t_idx/sample_res)] = W_target.reshape(N_recurrent_neurons)
            #theta_BCM_singles_samples[int(t_idx/sample_res)] = theta_BCM_single
            #theta_BCM_rec_samples[int(t_idx/sample_res)] = theta_BCM_rec.reshape(N_recurrent_neurons)
            weights_inh_single[int(t_idx/sample_res)] = w_IE_single
            weights_inh_rec[int(t_idx/sample_res)] = w_IE_rec.reshape(N_recurrent_neurons)
            weights_inh_target[int(t_idx/sample_res)] = w_IE_target.reshape(N_recurrent_neurons)

    results = {
        'weights_rec': weights_rec,
        'weights_target': weights_target,
        'rates_single': rates_single,
        'rates_rec': rates_rec,
        'rates_TD': rates_TD,
        'rates_target': rates_target,
        'rates_inh': rates_inh,
        'weights_inh_single': weights_inh_single,
        'weights_inh_rec': weights_inh_rec,
        'weights_inh_target': weights_inh_target,
        #'theta_BCM_singles': theta_BCM_singles_samples,
        'W_rec_plastic': W_rec_plastic,
        #'thetas_BCM_rec': theta_BCM_rec_samples,
        'sim_pars': pars
    }

    if pars['record_stim_responses']:
        results['presented_stims'] = presented_stims
        results['presented_reward'] = presented_rewards
        results['presented_stim_rec_rates'] = presented_stim_rec_rates

    return results

def plot_results(sim_results,title):
    fig,axes = plt.subplots(2,2,figsize=(10,10))
    axes[0][0].plot(sim_results['rates_single'],'--')
    axes[0][0].plot(sim_results['rates_target'])
    #axes[0][0].legend(labels=['single','target','same orientation'],loc='lower right')

    #axes[0][1].pcolor(sim_results['weights_target'])

    axes[1][1].plot(sim_results['weights_target'])
    #axes[1][1].legend(labels=['target','same orientation'],loc='lower right')

    axes[1][0].plot(sim_results['weights_inh_single'],'--')
    axes[1][0].plot(sim_results['weights_inh_target'])
    fig.suptitle(title,fontsize=20)

def plot_summary_results_quad(sim_results,title,savefig=False):
    fig,axes = plt.subplots(2,2,figsize=(10,10))
    axes[0][0].plot(sim_results[0]['weights_target'])
    axes[0][0].set_title('c_single=0.1,c_target=0.1')
    axes[0][1].plot(sim_results[1]['weights_target'])
    axes[0][1].set_title('c_single=0.1,c_target=1.0')
    axes[1][0].plot(sim_results[2]['weights_target'])
    axes[1][0].set_title('c_single=1.0,c_target=0.1')
    axes[1][1].plot(sim_results[3]['weights_target'])
    axes[1][1].set_title('c_single=1.0,c_target=1.0')
    axes[0][0].legend(['target','','same',''])

    fig.suptitle(title,fontsize=20)

    if savefig:
        plt.savefig('/home/ysweeney/Dropbox/Reports/top_down_learning/'+title+'.png')

def plot_summary_results_selectivity_evolution(sim_results,sim_pars,c_range,c_target_range,title,savefig=False):
    fig,axes = plt.subplots(1,len(c_target_range),figsize=(10*len(c_target_range),10))

    #sim_results.reverse()
    count = 0

    for c_trial_idx in xrange(len(c_range)):
        for c_rec_trial_idx in xrange(len(c_target_range)):
            sim_result = sim_results[count]
            count += 1
            selectivity = np.zeros(len(sim_result['weights_target']))
            for i in xrange(len(selectivity)):
                selectivity[i] = np.dot(np.cos(sim_pars['phi_rec']),sim_result['weights_target'][i])
            if len(c_target_range)>1:
                axes[c_rec_trial_idx].plot(selectivity,label=str(c_range[c_trial_idx]))
            else:
                axes.plot(selectivity,label=str(c_range[c_trial_idx]))

    try:
        for axis_idx in xrange(len(axes)):
            axes[axis_idx].legend()
            axes[axis_idx].set_title(c_target_range[axis_idx])
    except:
        axes.legend()
        axes.set_title(c_target_range[0])

    fig.suptitle(title,fontsize=20)

    if savefig:
        plt.savefig('/home/ysweeney/Dropbox/Reports/top_down_learning/'+title+'_reward_selectivity_evolution.png')
    else:
        plt.show()

def plot_summary_results_weight_evolution(sim_results,sim_pars,c_range,c_target_range,title,savefig=False):
    fig,axes = plt.subplots(1,len(c_target_range),figsize=(10*len(c_target_range),10))

    #sim_results.reverse()
    colors = sns.color_palette()
    count = 0

    for c_trial_idx in xrange(len(c_range)):
        for c_rec_trial_idx in xrange(len(c_target_range)):
            sim_result = sim_results[count]
            weights = sim_results[count]['weights_target'].transpose()
            count += 1

            if len(c_target_range)>1:
                axes[c_rec_trial_idx].plot(weights[0],'-',color=colors[c_trial_idx],label=str(c_range[c_trial_idx]))
                axes[c_rec_trial_idx].plot(weights[2],'--',color=colors[c_trial_idx],label=str(c_range[c_trial_idx]))
                axes[c_rec_trial_idx].plot(weights[3],'-*',color=colors[c_trial_idx],label=str(c_range[c_trial_idx]))
            else:
                axes.plot(weights[0],'-',color=colors[c_trial_idx],label=str(c_range[c_trial_idx]))
                axes.plot(weights[2],'--',color=colors[c_trial_idx],label=str(c_range[c_trial_idx]))
                axes.plot(weights[3],'-*',color=colors[c_trial_idx],label=str(c_range[c_trial_idx]))

    try:
        for axis_idx in xrange(len(axes)):
            axes[axis_idx].legend()
            axes[axis_idx].set_title(c_target_range[axis_idx])
    except:
        axes.legend()
        axes.set_title(c_target_range[0])

    fig.suptitle(title,fontsize=20)

    if savefig:
        plt.savefig('/home/ysweeney/Dropbox/Reports/top_down_learning/'+title+'_weights_evolution.png')
    else:
        plt.show()

def plot_summary_results_heatmap(sim_results,sim_pars,c_range,c_target_range,title,savefig=False,N_trials=1):
    fig,axes = plt.subplots(1,2,figsize=(20,10))

    #sim_results.reverse()
    colors = sns.color_palette('deep',len(c_range))
    count = 0

    c_sweep_devel_mean = np.zeros((len(c_range),len(c_target_range)))
    c_sweep_devel_std = np.zeros((len(c_range),len(c_target_range)))
    c_sweep_learn_mean = np.zeros((len(c_range),len(c_target_range)))
    c_sweep_learn_std = np.zeros((len(c_range),len(c_target_range)))

    for c_trial_idx in xrange(len(c_range)):
        for c_rec_trial_idx in xrange(len(c_target_range)):
            temp_devel_select = []
            temp_learn_select = []
            for trial_idx in xrange(N_trials):
                sim_result = sim_results[count]
                weights = sim_results[count]['weights_target']
                count += 1
                #temp_devel_select = np.dot(np.cos(sim_pars['phi_rec']-np.pi),weights[int(sim_pars['T_reward_start']/sim_pars['sample_res'])])
                #temp_learn_select = np.dot(np.cos(sim_pars['phi_rec']-0.0),weights[-1])
                temp_devel_select.append(weights[int(sim_pars['T_reward_start']/sim_pars['sample_res'])][2]-weights[int(sim_pars['T_reward_start']/sim_pars['sample_res'])][0])
                temp_learn_select.append(weights[-1][0]-weights[-1][2])

            c_sweep_devel_mean[c_trial_idx,c_rec_trial_idx] = np.mean(temp_devel_select)
            c_sweep_learn_mean[c_trial_idx,c_rec_trial_idx] = np.mean(temp_learn_select)
            c_sweep_devel_std[c_trial_idx,c_rec_trial_idx] = np.std(temp_devel_select)
            c_sweep_learn_std[c_trial_idx,c_rec_trial_idx] = np.std(temp_learn_select)

    sns.heatmap(c_sweep_devel_mean.transpose(),ax=axes[0],annot=False,xticklabels=c_range,yticklabels=c_target_range)
    sns.heatmap(c_sweep_learn_mean.transpose(),ax=axes[1],annot=False,xticklabels=c_range,yticklabels=c_target_range)
    axes[0].invert_yaxis()
    axes[1].invert_yaxis()
    axes[0].set_title('Feedforward selectivity',fontsize=20)
    axes[1].set_title('Reward selectivity',fontsize=20)
    axes[0].set_xlabel('student coupling',fontsize=20)
    axes[0].set_ylabel('teacher coupling',fontsize=20)
    axes[1].set_xlabel('student coupling',fontsize=20)
    axes[1].set_ylabel('teacher coupling',fontsize=20)
    #axes[1].pcolor(c_sweep_learn_mean)
    fig.suptitle(title,fontsize=20)

    if savefig:
        plt.savefig('/home/ysweeney/Dropbox/Reports/top_down_learning/'+title+'_heatmap.png')
    else:
        plt.show()

    fig,axes = plt.subplots(1,2,figsize=(20,10))

    sns.heatmap(c_sweep_devel_std.transpose(),ax=axes[0],annot=True,xticklabels=c_range,yticklabels=c_target_range)
    sns.heatmap(c_sweep_learn_std.transpose(),ax=axes[1],annot=True,xticklabels=c_range,yticklabels=c_target_range)
    axes[0].invert_yaxis()
    axes[1].invert_yaxis()
    axes[0].set_title('Feedforward selectivity',fontsize=20)
    axes[1].set_title('Reward selectivity',fontsize=20)
    axes[0].set_xlabel('student coupling',fontsize=20)
    axes[0].set_ylabel('teacher coupling',fontsize=20)
    axes[1].set_xlabel('student coupling',fontsize=20)
    axes[1].set_ylabel('teacher coupling',fontsize=20)
    #axes[1].pcolor(c_sweep_learn_mean)
    fig.suptitle(title,fontsize=20)

    if savefig and N_trials > 1:
        plt.savefig('/home/ysweeney/Dropbox/Reports/top_down_learning/'+title+'_heatmap_std.png')
    else:
        plt.show()

def plot_results_diverse_learning_rates(sim_results,title):
    sim_pars = sim_results['sim_pars']

    fig,axes = plt.subplots(2,3,figsize=(15,10))
    axes[0][0].plot(sim_results['rates_rec'])
    #axes[0][0].legend(labels=['single','target','same orientation'],loc='lower right')

    axes[0][1].pcolor(sim_results['W_rec_plastic'])
    axes[0][1].set_xlabel('presynaptic index')
    axes[0][1].set_ylabel('postsynaptic index')
    axes[0][1].set_title('after learning')

    W_sort = sim_results['weights_rec'][int(sim_pars['T_reward_start']/sim_pars['sample_res'])].reshape(sim_pars['N_recurrent_neurons'],sim_pars['N_recurrent_neurons']).copy()
    sort_idx = np.argsort(sim_pars['alpha_i'].flatten())
    W_sort = W_sort[:,sort_idx]
    W_sort = W_sort[sort_idx,:]

    #axes[1][2].pcolor(W_sort)
    #axes[1][2].set_xlabel('presynaptic index')
    #axes[1][2].set_ylabel('postsynaptic index')
    #axes[1][2].set_title('after development, sorted by alpha')


    axes[0][2].pcolor(sim_results['weights_rec'][int(sim_pars['T_reward_start']/sim_pars['sample_res'])].reshape(sim_pars['N_recurrent_neurons'],sim_pars['N_recurrent_neurons']))
    axes[0][2].set_xlabel('presynaptic index')
    axes[0][2].set_ylabel('postsynaptic index')
    axes[0][2].set_title('after development')

    for i in xrange(1,4):
        axes[1][1].plot(sim_results['weights_rec'][:,i])
    for i in xrange(32,36):
        axes[1][1].plot(sim_results['weights_rec'][:,i],'--')
    for i in xrange(64,68):
        axes[1][1].plot(sim_results['weights_rec'][:,i],'-.')
    axes[1][1].legend(labels=['target','same orientation'],loc='lower right')

    axes[1][0].plot(sim_results['weights_inh_rec'])
    fig.suptitle(title,fontsize=20)

    #axes[1][2].plot(sim_results['thetas_BCM_rec'])
    #axes[1][2].set_title('theta_BCM')

def plot_results_diverse_learning_rates_only_W(sim_results,title):
    sim_pars = sim_results['sim_pars']

    fig,axes = plt.subplots(1,2,figsize=(10,5))

    W_max = max(np.max(sim_results['weights_rec'][int(sim_pars['T_reward_start']/sim_pars['sample_res'])]),np.max(sim_results['W_rec_plastic']))
    axes[1].pcolor(sim_results['W_rec_plastic'])#,vmin=0,vmax=W_max)
    axes[1].set_xlabel('presynaptic index')
    axes[1].set_ylabel('postsynaptic index')
    axes[1].set_title('after learning')


    axes[0].pcolor(sim_results['weights_rec'][int(sim_pars['T_reward_start']/sim_pars['sample_res'])].reshape(sim_pars['N_recurrent_neurons'],sim_pars['N_recurrent_neurons']))#,vmin=0,vmax=W_max)
    axes[0].set_xlabel('presynaptic index')
    axes[0].set_ylabel('postsynaptic index')
    axes[0].set_title('after development')

    fig.suptitle(title,fontsize=20)

def get_mutual_information_individual_neurons(sim_results,use_fraction=0.5):
    from sklearn import metrics

    L = len(sim_results['presented_stims'])
    MI = []
    for i in xrange(48):
        MI.append(metrics.mutual_info_score(sim_results['presented_stims'][0.5*L:],sim_results['presented_stim_rec_rates'][0.5*L:,i]))

    return MI


def get_perceptron_score(sim_results,idx,start_at=0.5,N_samples=50,testing_reward=False,plot_trials=False,test_on_segment=None):
    from sklearn.linear_model import Perceptron
    P = Perceptron()

    responses = np.array(sim_results['presented_stim_rec_rates'])
    if testing_reward:
        reward_present = np.zeros(responses.shape[0])
        for i in xrange(len(sim_results['presented_stims'])):
            reward_present[i] = sim_results['presented_reward'][i] == sim_results['presented_stims'][i]
        target = np.array(reward_present,dtype='str')
    else:
        target = np.array(sim_results['presented_stims'],dtype='str')
    L = len(target)

    responses = responses[:,idx]

    if testing_reward:
        from sklearn.metrics import roc_curve,roc_auc_score
        #score =  roc_auc_score(reward_present[int(L*start_at):int(L*start_at)+N_samples],np.mean(responses,axis=1)[int(L*start_at):int(L*start_at)+N_samples])
        score =  roc_auc_score(reward_present[int(L*start_at):],np.mean(responses,axis=1)[(L*start_at):])
    else:
        P.fit(responses[L*start_at:L*start_at+N_samples],target[L*start_at:L*start_at+N_samples])
        if test_on_segment == None:
            #print 'r',  responses[L*start_at+N_samples:]
            #print 't',  target[L*start_at+N_samples:]
            score = P.score(responses[L*start_at+N_samples:],target[L*start_at+N_samples:])
        else:
            score = P.score(responses[test_on_segment],target[test_on_segment])
    if plot_trials:
        if testing_reward:
            from sklearn.metrics import roc_curve,roc_auc_score
            #[fpr,tpr,thresh] = roc_curve(reward_present[int(L*start_at):int(L*start_at)+N_samples],np.array(predictions))
            [fpr,tpr,thresh] = roc_curve(reward_present[int(L*start_at):int(L*start_at)+N_samples],np.mean(responses,axis=1)[int(L*start_at):int(L*start_at)+N_samples])
            print roc_auc_score(reward_present[int(L*start_at):int(L*start_at)+N_samples],np.mean(responses,axis=1)[int(L*start_at):int(L*start_at)+N_samples])
            plt.plot(fpr, tpr)#, label='ROC curve (area = %0.2f)' % roc_auc[2])
            #plt.scatter(range(N_samples),reward_present[int(L*start_at):int(L*start_at)+N_samples],c='r',s=40)
        else:
            predictions = []
            for response_idx in range(int(L*start_at),int(L*start_at)+N_samples):
                predictions.append(float(P.predict(responses[response_idx].reshape(1,-1))[0]))
            plt.scatter(range(N_samples),target[int(L*start_at):int(L*start_at)+N_samples],c='r',s=40)
            plt.scatter(range(N_samples),predictions,c='b',s=20)
    return score

def plot_combined_perceptron_trials(perceptron_results):
    all_scores = np.zeros((len(perceptron_results),len(perceptron_results[0]['all'])))
    half_scores = np.zeros((len(perceptron_results),len(perceptron_results[0]['all'])))
    slow_scores = np.zeros((len(perceptron_results),len(perceptron_results[0]['all'])))
    fast_scores = np.zeros((len(perceptron_results),len(perceptron_results[0]['all'])))

    for i in xrange(len(perceptron_results)):
        all_scores[i] = perceptron_results[i]['all']
        half_scores[i] = perceptron_results[i]['half']
        slow_scores[i] = perceptron_results[i]['slow']
        fast_scores[i] = perceptron_results[i]['fast']

    #N_sample_range = range(10,200,10)+range(200,1000,50)
    N_sample_range = range(len(perceptron_results[0]['all']))
    plt.errorbar(N_sample_range,1-np.mean(half_scores,axis=0),np.std(half_scores,axis=0))
    plt.errorbar(N_sample_range,1-np.mean(slow_scores,axis=0),np.std(slow_scores,axis=0))
    plt.errorbar(N_sample_range,1-np.mean(fast_scores,axis=0),np.std(fast_scores,axis=0))

    plt.legend(['half','slow','fast'])
    plt.ylabel('Perceptron error')
    plt.xlabel('# training samples')
    plt.show()

def plot_combined_ROC_trials(perceptron_results):
    all_scores = np.zeros(len(perceptron_results))
    half_scores = np.zeros(len(perceptron_results))
    slow_scores = np.zeros(len(perceptron_results))
    fast_scores = np.zeros(len(perceptron_results))

    for i in xrange(len(perceptron_results)):
        all_scores[i] = perceptron_results[i]['all']
        half_scores[i] = perceptron_results[i]['half']
        slow_scores[i] = perceptron_results[i]['slow']
        fast_scores[i] = perceptron_results[i]['fast']

    #N_sample_range = range(10,200,10)+range(200,1000,50)
    plt.bar([0],[np.mean(half_scores,axis=0)],yerr=[np.std(half_scores,axis=0)],color='b')
    plt.bar([1],[np.mean(fast_scores,axis=0)],yerr=[np.std(fast_scores,axis=0)],color='g')
    plt.bar([2],[np.mean(slow_scores,axis=0)],yerr=[np.std(slow_scores,axis=0)],color='r')


    plt.legend(['half','fast','slow'])
    plt.ylabel('Perceptron error')
    plt.xlabel('# training samples')
    plt.show()

def plot_clustered_W(W,n_cluster=2):
    from sklearn.cluster import spectral_clustering

    clustered_list = spectral_clustering(W,n_cluster)
    arg_sort = clustered_list.argsort()
    W_sorted = W.copy()
    W_sorted = W_sorted[:,arg_sort]
    W_sorted = W_sorted[arg_sort,:]

    plt.pcolor(W_sorted,cmap='Greys')
    plt.colorbar()
    plt.xlabel('presynaptic index')
    plt.ylabel('postsynaptic index')
    plt.show()

def plot_connectivity_summary(W,slow_idx=None,fast_idx=None,alpha_i=None,N_orientations=4):
    fig,axes = plt.subplots(2,2,figsize=(20,20))
    if fast_idx == None:
        args_s = alpha_i.argsort(axis=0)
        args_s = args_s.flatten()
        fast_idx = args_s[int(W.shape[0]*0.5):]
        slow_idx = args_s[:int(W.shape[0]*0.5)]


    out_degrees_from_slow = np.sum(W[slow_idx,:],axis=0)
    out_degrees_from_fast = np.sum(W[fast_idx,:],axis=0)
    axes[0][0].scatter(out_degrees_from_slow[fast_idx],out_degrees_from_fast[fast_idx],color='b')
    axes[0][0].scatter(out_degrees_from_slow[slow_idx],out_degrees_from_fast[slow_idx],color='r')
    axes[0][0].set_xlabel('out_degree_to_slow')
    axes[0][0].set_ylabel('out_degree_to_fast')
    axes[0][0].legend(['fast','slow'])

    in_degrees_from_slow = np.sum(W[:,slow_idx],axis=1)
    in_degrees_from_fast = np.sum(W[:,fast_idx],axis=1)
    axes[0][1].scatter(in_degrees_from_slow[fast_idx],in_degrees_from_fast[fast_idx],color='b')
    axes[0][1].scatter(in_degrees_from_slow[slow_idx],in_degrees_from_fast[slow_idx],color='r')
    axes[0][1].set_xlabel('in_degree_from_slow')
    axes[0][1].set_ylabel('in_degree_from_fast')
    axes[0][1].legend(['fast','slow'])

    N_neurons = W.shape[0]
    selectivity = np.zeros(N_neurons)
    N_per_orientation = int(N_neurons/N_orientations)
    for i in xrange(N_orientations):
        same_idx = range(i*N_per_orientation,(i+1)*N_per_orientation)
        other_idx = list(set(range(N_neurons))-set(same_idx))
        for j in xrange(N_per_orientation):
            selectivity[i*N_per_orientation + j] = np.sum(W[i*N_per_orientation + j,same_idx])-np.sum(W[i*N_per_orientation + j,other_idx])

    axes[1][1].scatter(fast_idx,selectivity[fast_idx],color='b')
    axes[1][1].scatter(slow_idx,selectivity[slow_idx],color='r')
    axes[1][1].set_ylabel('selectivity')
    axes[1][1].legend(['fast','slow'])

    axes[1][0].pcolor(W)

    #plt.show()

def plot_diverse_rates_selectivity_evolution(results,N_orientations=4,N_neurons=48):
    W_t = results['weights_rec']
    rewards = results['presented_reward']
    reward_res = int(len(rewards)/len(W_t))
    N_per_orientation = N_neurons/N_orientations

    reward_selectivities = np.zeros((len(W_t),N_neurons))
    FF_selectivities = np.zeros((len(W_t),N_neurons))

    reward_ids = np.zeros(len(W_t))

    for t_idx in xrange(len(W_t)):
        W = W_t[t_idx]
        reward = rewards[reward_res*t_idx]
        reward_ID = N_orientations-np.where(results['sim_pars']['phi_rec']==reward)[0][0]/N_per_orientation
        reward_ID = (np.where(results['sim_pars']['phi_stimuli']==reward)[0][0] + 4)%8
        if reward_ID == N_orientations:
            reward_ID = 0
        FF_selectivity,reward_selectivity = get_selectivity(W.reshape(N_neurons,N_neurons),N_orientations,reward_ID)
        FF_selectivities[t_idx] = FF_selectivity
        reward_selectivities[t_idx] = reward_selectivity
        reward_ids[t_idx] = reward_ID

    return FF_selectivities, reward_selectivities, reward_ids

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
            selectivity[i*N_per_orientation + j] = ((N_orientations-1)*np.sum(W[i*N_per_orientation + j,same_idx])-np.sum(W[i*N_per_orientation + j,other_idx]))
            if reward_ID == i:
                reward_selectivity[i*N_per_orientation + j] = 0.0
            else:
                reward_selectivity[i*N_per_orientation + j] = ((N_orientations-1)*np.sum(W[i*N_per_orientation + j,reward_idx])-np.sum(W[i*N_per_orientation + j,other_idx]))
    return selectivity, reward_selectivity

def run_sim_diverse_learning_rates_frozen(W_rec,simpars_original,simpars_change={}):
    simpars_original.update(simpars_change)
    simpars_original['T_exc_plasticity_start'] = simpars_original['T']
    simpars_original['synaptic_scaling'] = False
    simpars_original['reward_modulated_learning'] = False

    simpars_original['set_W_rec_plastic'] = True
    simpars_original['W_rec_plastic_passed'] = W_rec

    return run_sim_diverse_learning_rates(simpars_original)

def get_population_coupling(res):
    pop_rate = np.mean(res['rates_rec'],axis=1)
    pop_coupling = []
    for i in xrange(res['rates_rec'].shape[1]):
        pop_coupling.append(np.corrcoef(pop_rate,res['rates_rec'][:,i])[0][1])

    return np.array(pop_coupling)

def get_stimulus_selectivity_from_frozen_W(W_rec,simpars_original,simpars_change={}):
    results_frozen = run_sim_diverse_learning_rates_frozen(W_rec,simpars_original,simpars_change)

    stim_response = np.zeros(results_frozen['W_rec_plastic'].shape[0])
    baseline_response = np.zeros(results_frozen['W_rec_plastic'].shape[0])

    stim_counts = np.zeros(results_frozen['W_rec_plastic'].shape[0])
    baseline_counts = np.zeros(results_frozen['W_rec_plastic'].shape[0])

    for stim_idx in xrange(len(results_frozen['presented_stims'])):
        stim_neurons = results_frozen['sim_pars']['phi_rec']==results_frozen['presented_stims'][stim_idx]
        baseline_neurons = np.invert(stim_neurons)
        stim_response[stim_neurons] += results_frozen['presented_stim_rec_rates'][stim_idx][stim_neurons]
        stim_counts[stim_neurons] += 1
        baseline_response[baseline_neurons] += results_frozen['presented_stim_rec_rates'][stim_idx][baseline_neurons]
        baseline_counts[baseline_neurons] += 1

    selectivities = (stim_response/stim_counts)/(baseline_response/baseline_counts)

    return selectivities

