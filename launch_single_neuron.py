import sys
sys.path.append('/home/ysweeney/Dropbox/notebooks/')
sys.path.append('/Users/yann/Dropbox/notebooks/')

import py_scripts_yann

import numpy as np
import itertools

from matplotlib import pyplot as plt
from scipy import stats

import seaborn as sns

import run_single_neuron

T = 250*1000
N_recurrent_neurons = 4

sim_pars = {
    'T':T,
    'T_exc_plasticity_start': 20000,
    'T_reward_start': T*0.5,
    'BCM_target':5.0,
    'theta_BCM_single': 0.0,
    'H_TD_reward_stimulus': 10.0,
    'W_rec_max': 4.0/N_recurrent_neurons,
    'W_rec_static': 4.0/N_recurrent_neurons,
    'H_FF_stimulus_rec': 10.0,
    'H_FF_stimulus': 10.0,
    'sigma_OU' : 0.0,
    'ind_sigma_OU': 0.0,
    'sample_res': 1000,
    'alpha': 50.0e-7,
    'alpha_slow_ratio': 0.1,
    'theta_BCM_dt': 2.5e-4,
    'decay_lambda': 1.0,#0.9900000,
    'w_TD_up': 0.0,
    'plastic_weights_indices': range(N_recurrent_neurons),
    'eta': 50e-6,
    'W_inh_min': -50.0,
    'c_rec': 0.0,
    'N_recurrent_neurons': N_recurrent_neurons,
    'w_IE_single': -2.0,
    'w_IE': -0.2,
    'w_EI': 0.2,
    'w_TD_down': 1.0,
    'w_TD_up': 0.0,
    'phi_rec': np.array([0.0,np.pi/2,np.pi,3*np.pi/2]),
    'sample_res': 1000
}

sim_title = '36sweep_1trial_ghost_alpha_50e-7_FFsingle_10_reward_10_noise_ind_0_glob_0_pure_Hebb_halfprob_broad_ext_reward_unchanged_FF'


#c_target_range = [0.1,0.25,0.5,0.75,1.0]
#c_target_range = [0.1,1.0]
#c_target_range = [0.0,0.25,1.0]
#c_target_range=  [0.0]
#c_target_range = [0.0,0.2,0.4,0.6,0.8,1.0]#,1.0,2.5]
c_target_range = [0.0,0.1,0.2,0.3,0.4,0.5]
#c_target_range = [0.0,0.25,0.5]
#c_target_range = [0.0,0.3,0.6,0.9]
#c_range = [0.1,0.25,0.5,0.75,1.0,1.5,2.0]
#c_range = [0.1,.25,0.5,1.0,2.5]
#c_range = [0.1,0.5,1.0,2.5]#,1.0,2.5]
#c_range = [0.1,0.5,0.9,1.4,1.6,2.0,2.4]#,1.0,2.5]
#c_range = [0.1,0.7,1.3,1.9]
c_range = [0.0,0.2,0.4,0.6,0.8,1.0]
#c_range = [0.1,0.5,1.0]
#c_range = [0.0,1.0]

N_trials = 1

trial_results = []

c_sweep_devel_mean = np.zeros((len(c_range),len(c_target_range)))
c_sweep_devel_std = np.zeros((len(c_range),len(c_target_range)))
c_sweep_learn_mean = np.zeros((len(c_range),len(c_target_range)))
c_sweep_learn_std = np.zeros((len(c_range),len(c_target_range)))

#try:
#    get_ipython().magic(u'matplotlib qt')
#except:
#    pass
#plt.ion()
#fig,axes = plt.subplots(1,3)

for c_trial_idx in xrange(len(c_range)):
    for c_rec_trial_idx in xrange(len(c_target_range)):

        sim_pars['c'] = c_range[c_trial_idx]
        sim_pars['c_target'] = c_target_range[c_rec_trial_idx]

        print 'running for c_target = ', sim_pars['c_target']
        print 'running for c = ', sim_pars['c']

        #temp_reward_FF_difference = []
        for i in xrange(N_trials):
            print 'trial # ', i
            trial_result = run_single_neuron.run_sim_clean(sim_pars)

            #run_single_neuron.plot_results(trial_result,'reward_stim 25, noise both 5, alpha=1e-7, c='+str(sim_pars['c'])+',c_target='+str(sim_pars['c_target']))
            #plt.show()

            #plt.cla()
            #axes[0].plot(sample_exc_weights[:j/sample_res])
            #axes[1].plot(sample_inh_weights[:j/sample_res])
            #axes[2].pcolor(W)
            #plt.draw()
            #plt.pause(0.001)

            trial_results.append(trial_result)

            temp_devel_select = np.dot(np.cos(sim_pars['phi_rec']-np.pi),trial_result['weights_target'][int(sim_pars['T_reward_start']/sim_pars['sample_res'])])
            temp_learn_select = np.dot(np.cos(sim_pars['phi_rec']-0.0),trial_result['weights_target'][-1])

        c_sweep_devel_mean[c_trial_idx,c_rec_trial_idx] = np.mean(temp_devel_select)
        c_sweep_learn_mean[c_trial_idx,c_rec_trial_idx] = np.mean(temp_learn_select)

py_scripts_yann.save_pickle_safe('/home/ysweeney/Dropbox/notebooks/topdown_learning/pkl_results/'+str(sim_title)+'.pkl',trial_results)


if N_trials < 2:
    run_single_neuron.plot_summary_results_selectivity_evolution(trial_results,sim_pars,c_range,c_target_range,str(sim_title),True)
    run_single_neuron.plot_summary_results_weight_evolution(trial_results,sim_pars,c_range,c_target_range,str(sim_title),True)
run_single_neuron.plot_summary_results_heatmap(trial_results,sim_pars,c_range,c_target_range,str(sim_title),True,N_trials)

plt.figure()
plt.pcolor(c_sweep_devel_mean)
plt.yticks(c_range)
plt.xticks(c_target_range)
plt.ylabel('c')
plt.xlabel('c_target')
plt.colorbar()
#plt.savefig('/home/ysweeney/Dropbox/Reports/top_down_learning/c_c_target_sweep_devel_eta_5e-6_alpha_1e-7_H_reward_20_both_noise_5_T_500s.png')

print c_sweep_devel_mean

plt.figure()
plt.pcolor(c_sweep_learn_mean)
plt.yticks(c_range)
plt.xticks(c_target_range)
plt.ylabel('c')
plt.xlabel('c_target')
plt.colorbar()
#plt.savefig('/home/ysweeney/Dropbox/Reports/top_down_learning/c_c_target_sweep_learn_eta_5e-6_alpha_1e-7_H_reward_20_both_noise_5_T_500s.png')

print c_sweep_learn_mean

#run_single_neuron.plot_summary_results_quad(trial_results,'summary',False)
#plt.show()


