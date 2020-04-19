import sys
sys.path.append('/home/ysweeney/Dropbox/notebooks/')
sys.path.append('/Users/yann/Dropbox/notebooks/')

import py_scripts_yann

import numpy as np
import itertools

from matplotlib import pyplot as plt
from scipy import stats

import seaborn as sns


from sacred import Experiment
ex = Experiment()

from sacred.observers import FileStorageObserver
try:
    ex.observers.append(FileStorageObserver.create('/mnt/DATA/ysweeney/data/topdown_learning/diverse_rates_net_runs'))
except:
    print 'no observers'

import cPickle
from tempfile import mkdtemp
import shutil
import os

try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass

import run_single_neuron
import run_single_neuron_associative_learning

import multiprocessing


@ex.config
def config():
    sim_pars = {}

    T = 5000*1000
    N_recurrent_neurons = 48
    N_orientations = 4
    N_per_orientation = int(N_recurrent_neurons/N_orientations)

    #alpha_range = np.logspace(np.log10(0.05),np.log10(5.0),6)
    #sim_pars['learning_rates_student'] = [0.05]*2+[0.1]*2+[0.5]*2 + [0.75]*2 + [1.0]*2 + [2.5]*2
    #sim_pars['learning_rates_student'] = [alpha_range[0]]*2+[alpha_range[1]]*2+[alpha_range[2]]*2 + [alpha_range[3]]*2 + [alpha_range[4]]*2 + [alpha_range[5]]*2
    #sim_pars['learning_rates_student'] = list(np.logspace(np.log10(0.1),np.log10(15.0),12))
    #learning_rates_teacher = list(np.sort([0.1,0.2,0.5,1.0,1.5,2.0]))
    #learning_rates_teacher = list(np.sort([0.02,0.05,0.2,0.5,1.0,2.0]*2))
    sim_pars['learning_rates_student'] = [0.1]*6 + [1.0]*6
    #learning_rates_student = list(np.sort([0.1,0.2,0.5,1.0,1.5,2.0]*1))
    #learning_rates_student = list(np.sort([1.0,1.0,1.0,1.0,1.0,1.0]*1))
    #learning_rates_student = list(np.sort([0.1,0.2,0.5,0.75,1.0,2.0]*2))

    sim_pars = {
        'reward_modulated_learning': False,
        'reward_present': False,
        'record_stim_responses': True,
        'T':T,
        'T_exc_plasticity_start': 20000,
        'T_reward_start': T*0.3,
        'T_reward_end': T*0.90,
        'reward_switch_time': 50*1000,
        'BCM_target':3.5,
        'inh_target':5.0,
        'theta_BCM_single': 7.5,
        'H_TD_reward_stimulus': 15.0,
        'W_rec_max': 5.0/N_recurrent_neurons,
        'W_rec_static': 1.0/N_recurrent_neurons,
        'H_FF_stimulus_rec': 4.0,
        'H_FF_baseline': 0.0,
        'H_FF_stimulus': 10.0,
        'sigma_OU' : 0.0,
        'ind_sigma_OU': 15.0,
        'ind_mean_OU': 0.0,
        'ind_tau_OU': 10.0,
        'alpha': 5.0e-7,
        'alpha_slow_ratio': 0.1,
        'theta_BCM_dt': 5.0e-6,
        'decay_lambda': 1.0,#0.9999600,
        'w_TD_up': 0.0,
        'plastic_weights_indices': range(N_recurrent_neurons),
        'eta': 1e-5,
        'W_inh_min': -50.0,
        'c_rec': 1.0,
        'N_recurrent_neurons': N_recurrent_neurons,
        'N_orientations': N_orientations,
        'w_IE_single': -2.0,
        'w_IE': -0.2,
        'w_EI': 0.2,
        'w_TD_down': 1.0,
        'w_TD_up': 0.0,
        #'phi_rec': np.array([0.0,0.0,np.pi*0.0,np.pi*0.0,np.pi*0.5,np.pi*0.5,np.pi*.5,np.pi*.5]),
        'phi_rec':  np.sort([0,np.pi*0.5,np.pi,np.pi*1.5]*N_per_orientation),
        #'phi_rec':  np.sort([0,np.pi*0.25,np.pi*0.5,np.pi*0.75,np.pi,np.pi*1.25,np.pi*1.5,np.pi*1.75]*N_per_orientation),
        #'phi_stimuli': np.arange(0,np.pi*2+np.pi/4,np.pi/4)[:-1],
        'phi_stimuli': np.arange(0,np.pi*2+np.pi/2,np.pi/2)[:-1],
        #'phi_stimuli': np.array([np.pi*0.0,np.pi*0.5]),
        'tuning_width': 1.0,
        'W_max_scale': 0.1*np.array([1.0,1.0,1.0,1.0]*N_per_orientation),
        'alpha_i': 0.5e-7*np.array(sim_pars['learning_rates_student']*N_orientations).flatten().reshape(N_recurrent_neurons,1),
        'synaptic_scaling': True,
        'outgoing_scaling': False,
        'scaling_rate': 2e-4,
        'scaling_rates_i': 2.0e-5*np.ones(N_recurrent_neurons).flatten().reshape(N_recurrent_neurons,1),
        'scaling_target': .75,
        'sample_res': 10000,
        'random_dynamic_input': False,
        'dynamic_input_switch_time': 500*1000,
        'set_W_rec_plastic': False,
        'sweep_ID': '',
        'save_outputs': True,
        'N_trials': 15,
        'N_cores': 15,
        'distribute_exps': True

    }

    sim_pars['sim_title'] = 'broad_tuning_scaling_rate_2e-5_Wmax_5_8_orientatations_4_stimuli_48N_2_rates_p1_1_ind_OU_15'
    sim_pars['W_rec_max'] = (2.0/N_recurrent_neurons)*(sim_pars['scaling_target']/0.75)

    sim_pars['decoding_pars'] = {}
    sim_pars['decoding_pars']['W_scale'] = 1.0
    sim_pars['decoding_pars']['avg_weight'] = True
    sim_pars['decoding_pars']['shuffle_weight'] = False
    sim_pars['decoding_pars']['ind_OU'] = True
    sim_pars['decoding_pars']['ind_sigma_OU'] = 1.0
    sim_pars['decoding_pars']['H_FF_stimulus_rec'] = 8.0
    sim_pars['decoding_pars']['decoding_suffix'] = '_decoding_glob_OU_FF_avg_weight_logrange_15grid_indOU_1'
    sim_pars['decoding_pars']['launch_decoding_from_dir'] = os.getcwd()
    sim_pars['decoding_pars']['OU_range'] = np.logspace(np.log10(0.25),np.log10(20.0),15) #np.arange(0.0,21.1,3.0)#[0.0,1.0]#,5.0]
    sim_pars['decoding_pars']['FF_range'] = np.logspace(np.log10(0.1),np.log10(15.0),15) # np.arange(1.0,20.1,2.0)#[0.5,1.0.]#,5.0,7.5]

    sim_pars['H_FF_range'] = [4.0]
    #sim_pars['H_FF_range'] = [3.0,4.0,5.0]

    #OU_std_range  = [2.5,5.0,7.5]#,10.0,12.5,15.0]
    #OU_std_range  = [2.0,3.0,4.0,5.0,6.0,7.0]#,7.0]
    #OU_std_range  = [2.0,6.0,10.0,14.0,18.0]#,7.0]
    #H_FF_range = [3.0,4.0,5.0]
    #H_FF_range = [2.5,5.0,7.5]#,10.0,12.5,15.0]
    #sim_pars['scaling_target_range'] = [0.55,0.65,0.75,0.85]
    #sim_pars['scaling_target_range'] = [0.75,0.90,1.0,1.0,1.25,1.5]
    sim_pars['scaling_target_range'] = [0.75]
    #OU_std_range = scaling_target_range
    #OU_std_range  = [0.0,3.0,6.0,9.0,12.0,15.0,18.0,21.0,24.0]#,7.0]
    #H_FF_range = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]

    #OU_std_range = [2.0,10.0]
    #H_FF_range = [4.0,6.0]

    trial_results = []

    perceptron_results = []
    ROC_reward_results = []

#perceptron_diff_mean = np.zeros((len(OU_std_range),len(H_FF_range)))
#perceptron_diff_std= np.zeros((len(OU_std_range),len(H_FF_range)))
#ROC_diff_mean = np.zeros((len(OU_std_range),len(H_FF_range)))
#ROC_diff_std = np.zeros((len(OU_std_range),len(H_FF_range)))

#selectivity_diff = np.zeros((len(OU_std_range),len(H_FF_range)))
#reward_selectivity_diff = np.zeros((len(OU_std_range),len(H_FF_range)))
#selectivity_fast = np.zeros((len(OU_std_range),len(H_FF_range)))
#reward_selectivity_fast = np.zeros((len(OU_std_range),len(H_FF_range)))
#selectivity_slow = np.zeros((len(OU_std_range),len(H_FF_range)))
#reward_selectivity_slow = np.zeros((len(OU_std_range),len(H_FF_range)))

#try:
#    get_ipython().magic(u'matplotlib qt')
#except:
#    pass
#plt.ion()
#fig,axes = plt.subplots(1,3)

def run_iter(iter_pars):
    print iter_pars
    trial_results = []
    perceptron_results = []
    ROC_reward_results = []
    for i in xrange(N_trials):
        print 'trial # ', i
        #trial_result = run_single_neuron.run_sim_diverse_learning_rates(iter_pars)
        trial_result = run_single_neuron_associative_learning.run_sim_diverse_learning_rates(iter_pars)
        trial_results.append(trial_result)

        args_s = sim_pars['alpha_i'].argsort(axis=0)
        args_s = args_s.flatten()
        fast_idx = args_s[int(N_recurrent_neurons*0.5):]
        slow_idx = args_s[:int(N_recurrent_neurons*0.5)]

        idx = range(N_recurrent_neurons)
        half_idx = range(N_recurrent_neurons)[::2]
        quarter_idx = range(N_recurrent_neurons)[::4]

        all_score = []
        half_score = []
        quarter_score = []
        slow_score = []
        fast_score = []

        N_sample_start_range = np.arange(0.05,0.95,0.05)
        N_samples = 200 #range(10,200,10)+range(200,1000,50)
        for N_idx in N_sample_start_range:
            print 'N_idx ', N_idx
            all_score.append(run_single_neuron.get_perceptron_score(trial_result,idx,N_idx,N_samples))
            half_score.append(run_single_neuron.get_perceptron_score(trial_result,half_idx,N_idx,N_samples))
            quarter_score.append(run_single_neuron.get_perceptron_score(trial_result,quarter_idx,N_idx,N_samples))
            fast_score.append(run_single_neuron.get_perceptron_score(trial_result,fast_idx,N_idx,N_samples))
            slow_score.append(run_single_neuron.get_perceptron_score(trial_result,slow_idx,N_idx,N_samples))

        perceptron_result = {'all':all_score,'half':half_score,'quarter':quarter_score,'fast':fast_score,'slow':slow_score}
        perceptron_results.append(perceptron_result)

#        ROC_reward_result = {
#            'all': run_single_neuron.get_perceptron_score(trial_result,idx,float(iter_pars['T_reward_start']/sim_pars['T']),2,True,False),
#            'half': run_single_neuron.get_perceptron_score(trial_result,half_idx,float(iter_pars['T_reward_start']/sim_pars['T']),2,True,False),
#            'quarter': run_single_neuron.get_perceptron_score(trial_result,quarter_idx,float(iter_pars['T_reward_start']/sim_pars['T']),2,True,False),
#            'fast': run_single_neuron.get_perceptron_score(trial_result,fast_idx,float(iter_pars['T_reward_start']/sim_pars['T']),2,True,False),
#            'slow': run_single_neuron.get_perceptron_score(trial_result,slow_idx,float(iter_pars['T_reward_start']/sim_pars['T']),2,True,False)
#        }
#
#        ROC_reward_results.append(ROC_reward_result)

        #plt.figure()
        #plt.plot(N_sample_range,all_score)
        #plt.plot(N_sample_range,half_score)
        #plt.plot(N_sample_range,quarter_score)
        #plt.plot(N_sample_range,fast_score)
        #plt.plot(N_sample_range,slow_score)
        #plt.legend(['all','half','quarter','fast','slow'])
        #plt.xlabel('Nsamples')
        #plt.ylabel('perceptron score')

        #try:
        #    plt.savefig('/home/ysweeney/Dropbox/Reports/top_down_learning/'+sim_title+'_perceptron.png')
        #except:
        #    plt.savefig('/Users/yann/Dropbox/Reports/top_down_learning/'+sim_title+'_perceptron.png')

        print 'slow P score ', run_single_neuron.get_perceptron_score(trial_result,slow_idx,0.5,100)
        print 'fast P score ', run_single_neuron.get_perceptron_score(trial_result,fast_idx,0.5,100)

#        print 'ROC score ', ROC_reward_result

        #run_single_neuron.plot_results_diverse_learning_rates(trial_result,sim_title)
        #plt.show()
    if save_outputs:
        #py_scripts_yann.save_pickle_safe('/Users/yann/Desktop/'+str(sim_title) + str(iter_pars['sweep_ID'])+'.pkl',trial_results)
        #py_scripts_yann.save_pickle_safe('/Users/yann/Desktop/'+str(sim_title) + str(iter_pars['sweep_ID'])+'_perceptron_scores.pkl',perceptron_results)
        #py_scripts_yann.save_pickle_safe('/Users/yann/Desktop/'+str(sim_title) + str(iter_pars['sweep_ID'])+'_ROC_reward.pkl',ROC_reward_results)
        #try:
        #    py_scripts_yann.save_pickle_safe('~/Desktop/'+str(sim_title) + str(iter_pars['sweep_ID'])+'.pkl',trial_results)
        #    py_scripts_yann.save_pickle_safe('~/Desktop/'+str(sim_title) + str(iter_pars['sweep_ID'])+'_perceptron_scores.pkl',perceptron_results)
        #    py_scripts_yann.save_pickle_safe('~/Desktop/'+str(sim_title) + str(iter_pars['sweep_ID'])+'_ROC_reward.pkl',ROC_reward_results)
        #except:
        try:
            try:
                import os
                os.mkdir('/mnt/DATA/ysweeney/data/topdown_learning/'+str(sim_title))
            except:
                pass
            py_scripts_yann.save_pickle_safe('/mnt/DATA/ysweeney/data/topdown_learning/'+str(sim_title) +'/'+str(iter_pars['sweep_ID'])+'.pkl',trial_results)
            py_scripts_yann.save_pickle_safe('/mnt/DATA/ysweeney/data/topdown_learning/'+str(sim_title)+'/'+str(iter_pars['sweep_ID'])+'_perceptron_scores.pkl',perceptron_results)
            py_scripts_yann.save_pickle_safe('/mnt/DATA/ysweeney/data/topdown_learning/'+str(sim_title)+'/'+str(iter_pars['sweep_ID'])+'_ROC_reward.pkl',ROC_reward_results)
        except:
            try:
                py_scripts_yann.save_pickle_safe('/home/ysweeney/data/top_down_learning/'+str(sim_title)+str(iter_pars['sweep_ID'])+'.pkl',trial_results)
                py_scripts_yann.save_pickle_safe('/home/ysweeney/data/top_down_learning/'+str(sim_title)+str(iter_pars['sweep_ID'])+'_perceptron_scores.pkl',perceptron_results)
                py_scripts_yann.save_pickle_safe('/home/ysweeney/data/top_down_learning/'+str(sim_title)+str(iter_pars['sweep_ID'])+'_ROC_reward.pkl',ROC_reward_results)
            except:
                pass

        #try:
        #    plt.savefig('/home/ysweeney/Dropbox/Reports/top_down_learning/'+str(sim_title)+str(iter_pars['sweep_ID'])+'.png')
        #except:
        #    plt.savefig('/mnt/DATA/ysweeney/data/topdown_learning/'+str(sim_title)+str(iter_pars['sweep_ID'])+'.png')

        return trial_results


def run_iter_distributed(sim_pars,i):
    iter_pars = sim_pars.copy()

    iter_results = []
    for j in xrange(sim_pars['N_trials']):
        print 'trial # ', j
        #trial_result = run_single_neuron.run_sim_diverse_learning_rates(iter_pars)
        trial_result = run_single_neuron_associative_learning.run_sim_diverse_learning_rates(iter_pars)

        iter_pars_decoding = iter_pars.copy()
        iter_pars_decoding.update(sim_pars['decoding_pars'])
        #run_single_neuron_associative_learning.measure_decoding_from_frozen(iter_pars_decoding)
        trial_result['decoding_results'] = run_single_neuron_associative_learning.multiple_decoding_measures(iter_pars_decoding,sim_pars['decoding_pars']['W_scale']*trial_result['weights_rec'],3000,10,200000)
        #trial_result['decoding_results'] = run_single_neuron_associative_learning.multiple_decoding_measures_FF_OU_sweep(iter_pars_decoding,sim_pars['decoding_pars']['W_scale']*trial_result['weights_rec'],3000,10,200000)
        iter_results.append(trial_result)

    exp_dir = mkdtemp(dir="./")
    # create a filename for storing some data
    data_file = os.path.join(exp_dir,str(sim_pars['sim_title'])+'_'+str(sim_pars['sweep_ID'])+'.pkl')
    # assume some random results
    with open(data_file, 'wb') as f:
            print("writing results")
            cPickle.dump(iter_results, f)
    # add the result as an artifact, note that the name here is important
    # as sacred otherwise will try to save to the oddly named tmp subdirectory created
    ex.add_artifact(data_file, name=os.path.basename(data_file))
    # at the very end of the run delete the temporary directory
    # sacred will have taken care of copying all the results files over to the run directoy

    #for j in xrange(sim_pars['N_trials']):
    #    print 'trial # ', j
    #    #trial_result = run_single_neuron.run_sim_diverse_learning_rates(iter_pars)
    #    anim_file = os.path.join(exp_dir,str(sim_pars['sim_title'])+'_'+str(i))
    #    with open(anim_file, 'wb') as f:
    #        run_single_neuron_associative_learning.make_W_evolution_animation(iter_results[i]['weights_rec'],anim_file,sim_pars['N_recurrent_neurons'])
    #    ex.add_artifact(anim_file, name=os.path.basename(data_file))

    shutil.rmtree(exp_dir)

    return iter_results


def run_decoding_measure_distributed(sim_pars,results_path):
    trial_results = py_scripts_yann.load_pickle(results_path)
    iter_results = []
    for j in xrange(len(trial_results)): #sim_pars['N_trials']):
        print 'trial # ', j
        #trial_result = run_single_neuron.run_sim_diverse_learning_rates(iter_pars)
        iter_pars_decoding = trial_results[j]['sim_pars']
        iter_pars_decoding.update(sim_pars['decoding_pars'])

        #run_single_neuron_associative_learning.measure_decoding_from_frozen(iter_pars_decoding)
        #multiple_decoding_results = run_single_neuron_associative_learning.multiple_decoding_measures(iter_pars_decoding,trial_results[j]['weights_rec'],3000,10,200000)
        trial_result = trial_results[j]
        #trial_result['decoding_results'] = run_single_neuron_associative_learning.multiple_decoding_measures(iter_pars_decoding,trial_results[j]['weights_rec'],3000,5,100000)
        trial_result['decoding_results'] = run_single_neuron_associative_learning.multiple_decoding_measures_FF_OU_sweep(iter_pars_decoding,sim_pars['decoding_pars']['W_scale']*trial_result['weights_rec'],10000,5,100000,sim_pars['decoding_pars']['OU_range'],sim_pars['decoding_pars']['FF_range'],np.random.choice([0.8,0.9,1.0]),0.8,sim_pars['decoding_pars']['avg_weight'],sim_pars['decoding_pars']['shuffle_weight'],sim_pars['decoding_pars']['ind_OU'])

        trial_result['decoding_pars'] = sim_pars
        iter_results.append(trial_result)

    exp_dir = mkdtemp(dir="./")
    # create a filename for storing some data
    data_file = os.path.join(exp_dir,results_path+sim_pars['decoding_pars']['decoding_suffix']) # os.path.join(exp_dir,str(sim_pars['sim_title'])+'_'+str(sim_pars['sweep_ID'])+'.pkl')
    # assume some random results
    with open(data_file, 'wb') as f:
            print("writing results")
            cPickle.dump(iter_results, f)
    # add the result as an artifact, note that the name here is important
    # as sacred otherwise will try to save to the oddly named tmp subdirectory created
    ex.add_artifact(data_file, name=os.path.basename(data_file))
    # at the very end of the run delete the temporary directory
    # sacred will have taken care of copying all the results files over to the run directoy

    #for j in xrange(sim_pars['N_trials']):
    #    print 'trial # ', j
    #    #trial_result = run_single_neuron.run_sim_diverse_learning_rates(iter_pars)
    #    anim_file = os.path.join(exp_dir,str(sim_pars['sim_title'])+'_'+str(i))
    #    with open(anim_file, 'wb') as f:
    #        run_single_neuron_associative_learning.make_W_evolution_animation(iter_results[i]['weights_rec'],anim_file,sim_pars['N_recurrent_neurons'])
    #    ex.add_artifact(anim_file, name=os.path.basename(data_file))

    shutil.rmtree(exp_dir)

    return iter_results


def plot_sweep_results():
    reward_selectivity_diff_early = np.zeros((len(OU_std_range),len(H_FF_range)))
    reward_selectivity_fast_early = np.zeros((len(OU_std_range),len(H_FF_range)))
    reward_selectivity_slow_early = np.zeros((len(OU_std_range),len(H_FF_range)))
    reward_selectivity_diff = np.zeros((len(OU_std_range),len(H_FF_range)))
    reward_selectivity_fast = np.zeros((len(OU_std_range),len(H_FF_range)))
    reward_selectivity_slow = np.zeros((len(OU_std_range),len(H_FF_range)))
    selectivity_diff_post_reward= np.zeros((len(OU_std_range),len(H_FF_range)))
    selectivity_slow_post_reward = np.zeros((len(OU_std_range),len(H_FF_range)))
    selectivity_fast_post_reward = np.zeros((len(OU_std_range),len(H_FF_range)))
    selectivity_diff = np.zeros((len(OU_std_range),len(H_FF_range)))
    selectivity_slow = np.zeros((len(OU_std_range),len(H_FF_range)))
    selectivity_fast = np.zeros((len(OU_std_range),len(H_FF_range)))

    #for c_trial_idx in xrange(len(OU_std_range)):
    for c_trial_idx in xrange(len(scaling_target_range)):
        for c_rec_trial_idx in xrange(len(H_FF_range)):
            #sim_pars['ind_sigma_OU'] = OU_std_range[c_trial_idx]
            sim_pars['scaling_target'] = scaling_target_range[c_trial_idx]
            sim_pars['H_FF_stimulus_rec'] = H_FF_range[c_rec_trial_idx]
            #sim_pars['H_FF_stimulus_rec'] = np.append(np.ones(N_recurrent_neurons/2),np.zeros(N_recurrent_neurons/2))*H_FF_range[c_rec_trial_idx]

            #sim_pars['ind_mean_OU'] = 5.0 - sim_pars['H_FF_stimulus_rec']

            print 'running for ind_sigma_OU = ', sim_pars['ind_sigma_OU']
            print 'running for H_FF = ', sim_pars['H_FF_stimulus_rec']
            print 'running for ind_mean_OU = ', sim_pars['ind_mean_OU']
            print 'running for scaling target= ', sim_pars['scaling_target']

            #sim_pars['sweep_ID'] = '_ind_sigma_OU_' + str(sim_pars['ind_sigma_OU']) + '_H_FF_' + str(sim_pars['H_FF_stimulus_rec'])
            sim_pars['sweep_ID'] = '_scaling_target' + str(sim_pars['scaling_target']) + '_H_FF_' + str(sim_pars['H_FF_stimulus_rec'])

            #sim_results_perceptron = py_scripts_yann.load_pickle('/mnt/DATA/ysweeney/data/topdown_learning/'+str(sim_title)+sim_pars['sweep_ID']+'_perceptron_scores.pkl')
            #sim_results_perceptron = py_scripts_yann.load_pickle('/mnt/DATA/ysweeney/data/topdown_learning/'+str(sim_title)+'/'+sim_pars['sweep_ID']+'_perceptron_scores.pkl')

#            post_dev = []
#            post_reward = []
#
#            for res in sim_results_perceptron:
#                post_dev.append(res['slow'][9]-res['fast'][9])
#                post_reward.append(res['slow'][17]-res['fast'][17])

            #perceptron_diff_mean[c_trial_idx,c_rec_trial_idx] = np.mean(post_reward)
            #perceptron_diff_std[c_trial_idx,c_rec_trial_idx] = np.std(post_dev)


            #sim_results = py_scripts_yann.load_pickle('/mnt/DATA/ysweeney/data/topdown_learning/'+str(sim_title)+sim_pars['sweep_ID']+'.pkl')
            sim_results = py_scripts_yann.load_pickle('/mnt/DATA/ysweeney/data/topdown_learning/'+str(sim_title)+'/'+sim_pars['sweep_ID']+'.pkl')
            args_s = sim_results[0]['sim_pars']['alpha_i'].argsort(axis=0)
            args_s = args_s.flatten()
            fast_idx = args_s[int(sim_results[0]['W_rec_plastic'].shape[0]*0.5):]
            slow_idx = args_s[:int(sim_results[0]['W_rec_plastic'].shape[0]*0.5)]
            for res in sim_results:
                presented_reward = res['presented_reward'][-1]
                reward_ID = np.where(res['sim_pars']['phi_rec']==presented_reward)[0][0]/N_per_orientation
                print 'presented reward ', presented_reward, reward_ID
                #selectivity,temp = run_single_neuron.get_selectivity(res['weights_rec'])
                selectivity,temp = run_single_neuron.get_selectivity(res['weights_rec'][int(0.25*len(res['weights_rec']))].reshape(N_recurrent_neurons,N_recurrent_neurons),N_orientations,reward_ID)
                selectivity_post_reward,temp = run_single_neuron.get_selectivity(res['weights_rec'][int(0.99*len(res['weights_rec']))].reshape(N_recurrent_neurons,N_recurrent_neurons),N_orientations,reward_ID)
                temp,reward_selectivity_early= run_single_neuron.get_selectivity(res['weights_rec'][int(0.92*len(res['weights_rec']))].reshape(N_recurrent_neurons,N_recurrent_neurons),N_orientations,4)
                temp,reward_selectivity= run_single_neuron.get_selectivity(res['W_rec_plastic'],N_orientations,4)
#                reward_times = np.arange(res['sim_pars']['T_reward_start']+res['sim_pars']['reward_switch_time'],res['sim_pars']['T']+res['sim_pars']['reward_switch_time'],res['sim_pars']['reward_switch_time'])*(1.0/res['sim_pars']['sample_res'])-1
#                time_factor = res['sim_pars']['sample_res']/100
#                reward_selectivities = np.zeros((len(reward_times),N_recurrent_neurons))
#                for reward_time_idx in xrange(len(reward_times)):
#                    reward_time = reward_times[reward_time_idx]
#                    presented_reward = res['presented_reward'][reward_time*time_factor]
#                    print 'presented reward ', presented_reward, reward_time*time_factor
#                    reward_ID = np.where(res['sim_pars']['phi_rec']==presented_reward)[0][0]/N_per_orientation
#                    print 'reward ', reward_ID
#                    temp,reward_selectivity = run_single_neuron.get_selectivity(res['weights_rec'][reward_time].reshape(N_recurrent_neurons,N_recurrent_neurons),N_orientations,reward_ID)
#                    reward_selectivities[reward_time_idx] = reward_selectivity
#           print 'fast ', np.mean(reward_selectivity[fast_idx])
#           print 'slow ',  np.mean(reward_selectivity[slow_idx])
                selectivity_diff[c_trial_idx,c_rec_trial_idx] = np.mean(selectivity[slow_idx])-np.mean(selectivity[fast_idx])
                reward_selectivity_diff[c_trial_idx,c_rec_trial_idx] = np.mean(reward_selectivity[fast_idx])-np.mean(reward_selectivity[slow_idx])
                selectivity_fast[c_trial_idx,c_rec_trial_idx] = np.mean(selectivity[fast_idx])
                reward_selectivity_fast[c_trial_idx,c_rec_trial_idx] = np.mean(reward_selectivity[fast_idx])
                selectivity_slow[c_trial_idx,c_rec_trial_idx] = np.mean(selectivity[slow_idx])
                reward_selectivity_slow[c_trial_idx,c_rec_trial_idx] = np.mean(reward_selectivity[slow_idx])

                selectivity_fast_post_reward[c_trial_idx,c_rec_trial_idx] = np.mean(selectivity_post_reward[fast_idx])
                selectivity_slow_post_reward[c_trial_idx,c_rec_trial_idx] = np.mean(selectivity_post_reward[slow_idx])
                selectivity_diff_post_reward[c_trial_idx,c_rec_trial_idx] = np.mean(selectivity_post_reward[slow_idx])-np.mean(selectivity_post_reward[fast_idx])

                reward_selectivity_fast_early[c_trial_idx,c_rec_trial_idx] = np.mean(reward_selectivity_early[fast_idx])
                reward_selectivity_slow_early[c_trial_idx,c_rec_trial_idx] = np.mean(reward_selectivity_early[slow_idx])
                reward_selectivity_diff_early[c_trial_idx,c_rec_trial_idx] = np.mean(reward_selectivity_early[fast_idx])-np.mean(reward_selectivity_early[slow_idx])

        #print np.mean(reward_selectivity_fast),np.mean(reward_selectivity_slow)
    #print perceptron_diff_mean

    import seaborn as sns

    fig,axes = plt.subplots(4,3,figsize=(30,20))

    sns.heatmap(selectivity_fast.transpose(),ax=axes[0][0],annot=False,xticklabels=OU_std_range,yticklabels=H_FF_range)
    sns.heatmap(reward_selectivity_fast.transpose(),ax=axes[1][0],annot=False,xticklabels=OU_std_range,yticklabels=H_FF_range)
    sns.heatmap(selectivity_slow.transpose(),ax=axes[0][1],annot=False,xticklabels=OU_std_range,yticklabels=H_FF_range)
    sns.heatmap(reward_selectivity_slow.transpose(),ax=axes[1][1],annot=False,xticklabels=OU_std_range,yticklabels=H_FF_range)
    sns.heatmap(reward_selectivity_diff.transpose(),ax=axes[1][2],annot=False,xticklabels=OU_std_range,yticklabels=H_FF_range)
    sns.heatmap(selectivity_diff.transpose(),ax=axes[0][2],annot=False,xticklabels=OU_std_range,yticklabels=H_FF_range)

    sns.heatmap(selectivity_diff_post_reward.transpose(),ax=axes[2][2],annot=False,xticklabels=OU_std_range,yticklabels=H_FF_range)
    sns.heatmap(selectivity_slow_post_reward.transpose(),ax=axes[2][1],annot=False,xticklabels=OU_std_range,yticklabels=H_FF_range)
    sns.heatmap(selectivity_fast_post_reward.transpose(),ax=axes[2][0],annot=False,xticklabels=OU_std_range,yticklabels=H_FF_range)

    sns.heatmap(reward_selectivity_slow_early.transpose(),ax=axes[3][1],annot=False,xticklabels=OU_std_range,yticklabels=H_FF_range)
    sns.heatmap(reward_selectivity_fast_early.transpose(),ax=axes[3][0],annot=False,xticklabels=OU_std_range,yticklabels=H_FF_range)
    sns.heatmap(reward_selectivity_diff_early.transpose(),ax=axes[3][2],annot=False,xticklabels=OU_std_range,yticklabels=H_FF_range)

    for axis in axes.flatten():
        axis.invert_yaxis()
        axis.set_xlabel('OU_std',fontsize=20)
        axis.set_ylabel('H_FF',fontsize=20)


    axes[0][0].set_title('FF selectivity (fast)',fontsize=20)
    axes[1][1].set_title('Reward selectivity (slow)',fontsize=20)
    axes[1][0].set_title('Reward selectivity (fast)',fontsize=20)
    axes[0][1].set_title('FF selectivity (slow)',fontsize=20)
    axes[0][2].set_title('FF selectivity (diff)',fontsize=20)
    axes[1][2].set_title('Reward selectivity (diff)',fontsize=20)
    axes[2][0].set_title('FF selectivity (fast)',fontsize=20)
    axes[3][1].set_title('Early Reward selectivity (slow)',fontsize=20)
    axes[3][0].set_title('Early Reward selectivity (fast)',fontsize=20)
    axes[2][1].set_title('Post-reward FF selectivity (slow)',fontsize=20)
    axes[2][2].set_title('Post-reward FF selectivity (diff)',fontsize=20)
    axes[3][2].set_title('Early Reward selectivity (diff)',fontsize=20)
    #axes[1].pcolor(c_sweep_learn_mean)
    fig.suptitle(sim_title,fontsize=20)

def plot_sweep_coupling(two_groups=True):
    from scipy import stats
    pop_coupling_r2_during_sim = np.zeros((len(OU_std_range),len(H_FF_range)))
    pop_coupling_r2_frozen = np.zeros((len(OU_std_range),len(H_FF_range)))
    pop_coupling_diff_during_sim = np.zeros((len(OU_std_range),len(H_FF_range)))
    pop_coupling_diff_frozen = np.zeros((len(OU_std_range),len(H_FF_range)))

    pop_coupling_r2_frozen_high_noise = np.zeros((len(OU_std_range),len(H_FF_range)))
    pop_coupling_diff_frozen_high_noise = np.zeros((len(OU_std_range),len(H_FF_range)))

    #for c_trial_idx in xrange(len(OU_std_range)):
    for c_trial_idx in xrange(len(scaling_target_range)):
        for c_rec_trial_idx in xrange(len(H_FF_range)):
            #sim_pars['ind_sigma_OU'] = OU_std_range[c_trial_idx]
            sim_pars['scaling_target'] = scaling_target_range[c_trial_idx]
            sim_pars['H_FF_stimulus_rec'] = H_FF_range[c_rec_trial_idx]

            #sim_pars['ind_mean_OU'] = 5.0 - sim_pars['H_FF_stimulus_rec']

            print 'running for ind_sigma_OU = ', sim_pars['ind_sigma_OU']
            print 'running for H_FF = ', sim_pars['H_FF_stimulus_rec']
            print 'running for ind_mean_OU = ', sim_pars['ind_mean_OU']
            print 'running for scaling target= ', sim_pars['scaling_target']

            #sim_pars['sweep_ID'] = '_ind_sigma_OU_' + str(sim_pars['ind_sigma_OU']) + '_H_FF_' + str(sim_pars['H_FF_stimulus_rec'])
            sim_pars['sweep_ID'] = '_scaling_target' + str(sim_pars['scaling_target']) + '_H_FF_' + str(sim_pars['H_FF_stimulus_rec'])

            #sim_results = py_scripts_yann.load_pickle('/mnt/DATA/ysweeney/data/topdown_learning/'+str(sim_title)+sim_pars['sweep_ID']+'.pkl')
            sim_results = py_scripts_yann.load_pickle('/mnt/DATA/ysweeney/data/topdown_learning/'+str(sim_title)+'/'+sim_pars['sweep_ID']+'.pkl')
            args_s = sim_results[0]['sim_pars']['alpha_i'].argsort(axis=0)
            args_s = args_s.flatten()
            fast_idx = args_s[int(sim_results[0]['W_rec_plastic'].shape[0]*0.5):]
            slow_idx = args_s[:int(sim_results[0]['W_rec_plastic'].shape[0]*0.5)]
            for res in sim_results:
                sim_pars_pass ={
                    'T': 100*1000,
                    'sample_res':1,
                    'H_FF_stimulus_rec': 5.0,
                    'ind_sigma_OU': 5.0
                }
                W_pass = res['weights_rec'][int(0.5*len(res['weights_rec']))].reshape(N_recurrent_neurons,N_recurrent_neurons)
                res_frozen = run_single_neuron.run_sim_diverse_learning_rates_frozen(W_pass,res['sim_pars'],sim_pars_pass)
                pop_coupling_frozen = run_single_neuron.get_population_coupling(res_frozen)
                pop_coupling_during_sim = run_single_neuron.get_population_coupling(res)

                if two_groups:
                    pop_coupling_r2_during_sim[c_trial_idx,c_rec_trial_idx] = stats.linregress(np.argsort(res['sim_pars']['alpha_i'].flatten()),pop_coupling_during_sim[np.argsort(res['sim_pars']['alpha_i'].flatten())])[2]
                    pop_coupling_r2_frozen[c_trial_idx,c_rec_trial_idx] = stats.linregress(np.argsort(res['sim_pars']['alpha_i'].flatten()),pop_coupling_frozen[np.argsort(res['sim_pars']['alpha_i'].flatten())])[2]
                    pop_coupling_diff_during_sim[c_trial_idx,c_rec_trial_idx] = np.mean(pop_coupling_during_sim[fast_idx])-np.mean(pop_coupling_during_sim[slow_idx])
                    pop_coupling_diff_frozen[c_trial_idx,c_rec_trial_idx] = np.mean(pop_coupling_frozen[fast_idx])-np.mean(pop_coupling_frozen[slow_idx])

                sim_pars_pass['H_FF_stimuls_rec'] = 2.0
                sim_pars_pass['ind_sigma_OU'] = 15.0
                W_pass = res['weights_rec'][int(0.5*len(res['weights_rec']))].reshape(N_recurrent_neurons,N_recurrent_neurons)
                res_frozen = run_single_neuron.run_sim_diverse_learning_rates_frozen(W_pass,res['sim_pars'],sim_pars_pass)
                pop_coupling_frozen = run_single_neuron.get_population_coupling(res_frozen)
                pop_coupling_during_sim = run_single_neuron.get_population_coupling(res)

                if two_groups:
                    pop_coupling_r2_frozen_high_noise[c_trial_idx,c_rec_trial_idx] = stats.linregress(np.argsort(res['sim_pars']['alpha_i'].flatten()),pop_coupling_frozen[np.argsort(res['sim_pars']['alpha_i'].flatten())])[2]
                    pop_coupling_diff_frozen_high_noise[c_trial_idx,c_rec_trial_idx] = np.mean(pop_coupling_frozen[fast_idx])-np.mean(pop_coupling_frozen[slow_idx])
                    print 'r^2, p = ', stats.linregress(np.argsort(res['sim_pars']['alpha_i'].flatten()),pop_coupling_frozen[np.argsort(res['sim_pars']['alpha_i'].flatten())])
    import seaborn as sns

    fig,axes = plt.subplots(2,3,figsize=(20,10))

    sns.heatmap(pop_coupling_diff_during_sim.transpose(),ax=axes[0][0],annot=False,xticklabels=OU_std_range,yticklabels=H_FF_range)
    sns.heatmap(pop_coupling_diff_frozen.transpose(),ax=axes[0][1],annot=False,xticklabels=OU_std_range,yticklabels=H_FF_range)
    sns.heatmap(pop_coupling_r2_during_sim.transpose(),ax=axes[1][0],annot=False,xticklabels=OU_std_range,yticklabels=H_FF_range)
    sns.heatmap(pop_coupling_r2_frozen.transpose(),ax=axes[1][1],annot=False,xticklabels=OU_std_range,yticklabels=H_FF_range)
    sns.heatmap(pop_coupling_diff_frozen_high_noise.transpose(),ax=axes[0][2],annot=False,xticklabels=OU_std_range,yticklabels=H_FF_range)
    sns.heatmap(pop_coupling_r2_frozen_high_noise.transpose(),ax=axes[1][2],annot=False,xticklabels=OU_std_range,yticklabels=H_FF_range)

    for axis in axes.flatten():
        axis.invert_yaxis()
        axis.set_xlabel('OU_std',fontsize=20)
        axis.set_ylabel('H_FF',fontsize=20)


    axes[0][0].set_title('population coupling difference, during sim',fontsize=20)
    axes[0][1].set_title('population coupling difference, from frozen',fontsize=20)
    axes[1][0].set_title('population coupling r^2, during sim',fontsize=20)
    axes[1][1].set_title('population coupling r^2, from frozen',fontsize=20)
    axes[0][2].set_title('population coupling diff, from frozen high noise',fontsize=20)
    axes[1][2].set_title('population coupling r^2, from frozen, high noise',fontsize=20)
    #axes[1].pcolor(c_sweep_learn_mean)
    fig.suptitle(sim_title,fontsize=20)

@ex.command
def launch_multiple_decoding_measures(sim_pars):
    from multiprocessing import Pool, Process, Manager
    import glob

    file_paths = glob.glob(sim_pars['decoding_pars']['launch_decoding_from_dir']+'/*.pkl')

    with Manager() as manager:
        #exp_results_list = manager.list()  # <-- can be shared between processes.
        processes = []
        for file_path in file_paths:
            print file_path
            p = Process(target=run_decoding_measure_distributed, args=(sim_pars,file_path))  # Passing the list
            np.random.seed()
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

def gather_multiple_decoding_measures(file_dir,save_str=None):
    print 'gathering decoding results ', file_dir, save_str
    import glob
    files = glob.glob(file_dir+'*.pkl*')
    #decoding_results = {'detect_1': [], 'pairs': [], 'stimA': [], 'stim_broad': []}
    decoding_results = {'pairs': [], 'stimA': [], 'stimA_alone': []}#, 'detect_1_alone': [], 'detect_1': []}

    for file_str in files:
        res_file = py_scripts_yann.load_pickle(file_str)
        for key in decoding_results.keys():
            for res_trial in res_file:
                try:
                    #decoding_results[key].append(np.mean(res_trial['decoding_results'][key],axis=2)[-1,-1])
                    decoding_results[key].append(np.mean(res_trial['decoding_results'][key],axis=2))
                except:
                    pass

    if save_str == None:
        return decoding_results
    else:
        py_scripts_yann.save_pickle(file_dir+save_str+'_decoding_results.pkl',decoding_results)

        import json
        mean_results = {}
        for key in decoding_results.keys():
            mean_results[key]=np.mean(decoding_results[key])
        json.dump(mean_results, open(file_dir+save_str+'_mean_decoding_results.txt','w'))
        return decoding_results

def plot_summary_multiple_decoding_measures(file_dir_fast,file_dir_slow,file_dir_mixed,save_str):
    print 'gathering decoding results ', save_str
    #import glob
    #files_fast = glob.glob(file_dir_fast+'*.pkl*')
    #files_slow = glob.glob(file_dir_slow+'*.pkl*')
    #files_mixed = glob.glob(file_dir_mixed+'*.pkl*')

    decoding_results_fast = gather_multiple_decoding_measures(file_dir_fast)
    decoding_results_slow = gather_multiple_decoding_measures(file_dir_slow)
    decoding_results_mixed = gather_multiple_decoding_measures(file_dir_mixed)

    #decoding_results_slow = {'pairs': [], 'stimA_alone': []}
    #decoding_results_mixed = {'pairs': [], 'stimA_alone': []}

    for key in decoding_results_fast.keys():
        fig,ax = plt.subplots(2,2)
        #vmax_k = np.max(decoding_results_slow_global_highrange_4_v2[key])
        #vmin_k = np.min(decoding_results_slow_global_highrange_4_v2[key])
        vmean_k = np.mean(np.mean(np.array([decoding_results_slow[key],decoding_results_fast[key],decoding_results_mixed[key]]),axis=0),axis=0)
        vmin_k =-0.2
        vmax_k = 0.2
        ax[0][0].pcolor((np.mean(np.array(decoding_results_fast[key]),axis=0)-vmean_k)/vmean_k,vmin=vmin_k,vmax=vmax_k,cmap='PuOr')
        ax[0][0].set_title('all fast')
        ax[1][0].pcolor((np.mean(np.array(decoding_results_slow[key]),axis=0)-vmean_k)/vmean_k,vmin=vmin_k,vmax=vmax_k,cmap='PuOr')
        ax[1][0].set_title('all slow')
        im = ax[0][1].pcolor((np.mean(np.array(decoding_results_mixed[key]),axis=0)-vmean_k)/vmean_k,vmax=vmax_k,vmin=vmin_k,cmap='PuOr')
        ax[0][1].set_title('mixed')
        im_2 = ax[1][1].pcolor(vmean_k,cmap='Greys')
        ax[1][1].set_title('absolute performance (mean)')
        cax = fig.add_axes([0.92, 0.1, 0.03, 0.8])
        cax_2 = fig.add_axes([0.03, 0.1, 0.03, 0.8])
        for ax_i in ax.flatten():
            ax_i.set_xlabel('FF input')
            ax_i.set_ylabel('noise')

        fig.colorbar(im, cax=cax)
        fig.colorbar(im_2, cax=cax_2)
        fig.suptitle(key)
        fig.savefig('/home/ysweeney/data/top_down_learning/plots/'+save_str+'_'+str(key)+'.pdf')


def main():
    #for c_trial_idx in xrange(len(OU_std_range)):
    for c_trial_idx in xrange(len(scaling_target_range)):
        for c_rec_trial_idx in xrange(len(H_FF_range)):

            #sim_pars['ind_sigma_OU'] = OU_std_range[c_trial_idx]
            sim_pars['scaling_target'] = scaling_target_range[c_trial_idx]
            # normalising W_max by Wmax=2 for scaling_target=0.75
            sim_pars['W_rec_max'] = (2.0/N_recurrent_neurons)*(scaling_target_range[c_trial_idx]/0.75)
            sim_pars['H_FF_stimulus_rec'] = H_FF_range[c_rec_trial_idx]
            #sim_pars['H_FF_stimulus_rec'] = np.append(np.zeros(N_recurrent_neurons/2),np.ones(N_recurrent_neurons/2))*H_FF_range[c_rec_trial_idx]

            #sim_pars['ind_mean_OU'] = 5.0 - sim_pars['H_FF_stimulus_rec']

            print 'running for ind_sigma_OU = ', sim_pars['ind_sigma_OU']
            print 'running for H_FF = ', sim_pars['H_FF_stimulus_rec']
            print 'running for ind_mean_OU = ', sim_pars['ind_mean_OU']
            print 'running for scaling target= ', sim_pars['scaling_target']
            print 'running for W_max= ', sim_pars['W_rec_max']

            #sim_pars['sweep_ID'] = '_ind_sigma_OU_' + str(sim_pars['ind_sigma_OU']) + '_H_FF_' + str(sim_pars['H_FF_stimulus_rec'])
            #sim_pars['sweep_ID'] = '_ind_sigma_OU_' + str(sim_pars['ind_sigma_OU']) + '_H_FF_' + str(H_FF_range[c_rec_trial_idx])
            sim_pars['sweep_ID'] = '_scaling_target' + str(sim_pars['scaling_target']) + '_H_FF_' + str(sim_pars['H_FF_stimulus_rec'])

            multiprocessing.Process(target=run_iter,args=[sim_pars]).start()


@ex.main
def auto_main(sim_pars):
    print 'running multicore'
    from multiprocessing import Pool, Process, Manager
    if sim_pars['distribute_exps']:
        with Manager() as manager:
            #exp_results_list = manager.list()  # <-- can be shared between processes.
            processes = []
            sim_pars['N_trials'] = int(sim_pars['N_trials']/sim_pars['N_cores'])
            for i in range(sim_pars['N_cores']):
                sim_pars['sweep_ID'] = i
                p = Process(target=run_iter_distributed, args=(sim_pars,i))  # Passing the list
                np.random.seed()
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            #py_scripts_yann.save_pickle_safe('/mnt/DATA/ysweeney/data/topdown_learning/'+str(sim_title)+'.pkl',exp_results_list)
        #p = Pool(N_cores)
        #par_sweep_key = ['N_sims']
        #p.map(run_iter,int(sim_pars['N_sims']/N_cores))
    else:
        with Manager() as manager:
            processes = []
            for c_trial_idx in xrange(len(sim_pars['scaling_target_range'])):
                for c_rec_trial_idx in xrange(len(sim_pars['H_FF_range'])):
                    iter_pars = sim_pars.copy()
                    #sim_pars['ind_sigma_OU'] = OU_std_range[c_trial_idx]
                    iter_pars['scaling_target'] = sim_pars['scaling_target_range'][c_trial_idx]
                    # normalising W_max by Wmax=2 for scaling_target=0.75
                    iter_pars['W_rec_max'] = (2.0/sim_pars['N_recurrent_neurons'])*(sim_pars['scaling_target_range'][c_trial_idx]/0.75)
                    iter_pars['H_FF_stimulus_rec'] = sim_pars['H_FF_range'][c_rec_trial_idx]
                    #sim_pars['H_FF_stimulus_rec'] = np.append(np.zeros(N_recurrent_neurons/2),np.ones(N_recurrent_neurons/2))*H_FF_range[c_rec_trial_idx]

                    #sim_pars['ind_mean_OU'] = 5.0 - sim_pars['H_FF_stimulus_rec']

                    print 'running for ind_sigma_OU = ', iter_pars['ind_sigma_OU']
                    print 'running for H_FF = ', iter_pars['H_FF_stimulus_rec']
                    print 'running for ind_mean_OU = ', iter_pars['ind_mean_OU']
                    print 'running for scaling target= ', iter_pars['scaling_target']
                    print 'running for W_max= ', iter_pars['W_rec_max']

                    #sim_pars['sweep_ID'] = '_ind_sigma_OU_' + str(sim_pars['ind_sigma_OU']) + '_H_FF_' + str(sim_pars['H_FF_stimulus_rec'])
                    #sim_pars['sweep_ID'] = '_ind_sigma_OU_' + str(sim_pars['ind_sigma_OU']) + '_H_FF_' + str(H_FF_range[c_rec_trial_idx])
                    iter_pars['sweep_ID'] = '_scaling_target' + str(sim_pars['scaling_target']) + '_H_FF_' + str(sim_pars['H_FF_stimulus_rec'])

                    #iter_pars[sim_pars['par_sweep_key']] = par_val

                    p = Process(target=run_iter_distributed, args=(iter_pars,0))#sim_pars['par_sweep_key']+str(iter_pars[sim_pars['par_sweep_key']])))  # Passing the list
                    np.random.seed()
                    p.start()
                    processes.append(p)
	    for p in processes:
		p.join()


        #p = Pool(len(sim_pars['par_sweep_vals']))
        #p.map(run_iter,par_sweep_vals)
        #p.map(func_star, itertools.izip(sim_pars['par_sweep_vals'], itertools.repeat(sim_pars)))


if __name__ == "__main__":
    import sys
    if len(sys.argv)> 1 and sys.argv[1] == 'decoding':
        if len(sys.argv)< 2:
            ex.run('launch_multiple_decoding_measures')
        else:
            'launch decoding with dir given', sys.argv[2]
            sim_pars_pass = {'sim_pars':{'decoding_pars':{'launch_decoding_from_dir':sys.argv[2]}}}
            ex.run('launch_multiple_decoding_measures',sim_pars_pass)
    elif len(sys.argv)> 1 and sys.argv[1] == 'decoding_gather':
        if len(sys.argv)>3 and type(sys.argv[2]) is str:
            gather_multiple_decoding_measures(sys.argv[2],sys.argv[3])
        else:
            import os
            gather_multiple_decoding_measures(os.getcwd(),sys.argv[2])
    else:
        ex.run()


