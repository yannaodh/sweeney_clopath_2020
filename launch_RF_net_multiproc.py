import sys
sys.path.append('/home/ysweeney/Dropbox/notebooks/')
sys.path.append('/Users/yann/Dropbox/notebooks/')

import py_scripts_yann

import numpy as np
import itertools

from matplotlib import pyplot as plt
from scipy import stats

import seaborn as sns

#import ipyparallel as ipp
import time

from sacred import Experiment
ex = Experiment()

from sacred.observers import FileStorageObserver
try:
    ex.observers.append(FileStorageObserver.create('/mnt/DATA/ysweeney/data/topdown_learning/RF_net_runs'))
except:
    try:
        ex.observers.append(FileStorageObserver.create('/media/ysweeney/HDD/topdown_learning/RF_net_runs'))
    except:
        print 'no observers'

import multiprocessing

import cPickle
from tempfile import mkdtemp
import shutil
import os

import run_RF_net
try:
    import mkl
    mkl.set_num_threads(1)
except:
    'Couldnt set mkl threads'
    pass


@ex.config
def config():
    eta_base = 0.001
    N = 250

    sim_pars = {
        'T':2500000,
        'sample_res':100000,
        'N_filters': N,
        'group_corr_size': 40,
        'group_corr_strength': 0.0,
        'N_per_group': int(N/10),
        'N_sims': 15,
        'pop_coupling_add_factor': 0.05,
        'pop_coupling_mult_factor': 1.0,
        'eta': eta_base*0.005,
        'stim_period': 100,
        'ext_OU_tau': 50.0,
        'ext_OU_sigma': 2.5,
        'OU_global': 0.0,
        'x_corr_mean': 0.0,
        'x_corr_std': 0.0,
        'W_max': 2.0/N,
        'W_scaling_target': 0.5,  #recurrent,inh and ext input should all be order 1 (dynamic range of neuron is ~ 5-15
        'scaling_rate': eta_base*0.25,
        'W_input': 2.5,
        'W_IE_init': 0.1,
        'W_EI': 2.0/N,
        'eta1_base': 0.000020*eta_base,
        'distribute_exps': True,
        'N_cores': 15,
        'uniform_prob_conn': True,
        'W_density': 1.0,
        'uniform_W_init': True,
        'prune_weights': False,
        'only_diverse': True,
        'prune_thresh': 1.0, #absolute weight value
        'prune_prob': 0.2, #absolute weight value
        'prune_freq': 2000,
        'prune_stop': 120000,
        'prune_start': 50000,
        'random_theta': False,
        'random_phi': False,
        'theta_jitter': 0.0,
        'pc_dist': 'uniform',
        'pc_bimodal_prob': 0.5, #higher = more fast neurons
        'alpha_range': 50.0,
        'homogenous_base': 5.0,
        'present_random_bars': False,
        'measure_responses': False,
        'heterogenous_r_0': False,
        'heterogenous_ext_OU_tau': False,
        'heterogenous_ext_OU_sigma': True,
        'r_0_range': 10.0,
    }


    #par_sweep_key = 'pop_coupling_add_factor'
    #par_sweep_vals = [0.01,0.02,0.05,0.1,0.2]
    #par_sweep_key = 'pop_coupling_mult_factor'
    #par_sweep_vals = [0.1,0.25,0.5,1.0,1.5]

    #par_sweep_key = 'scaling_rate'
    #par_sweep_vals = [eta1_base*1.0,eta1_base*5.0,eta1_base*10.0,eta1_base*20.0]
    #par_sweep_key = 'scaling_target'
    #par_sweep_vals = [100,250,500]

    sim_pars['par_sweep_key'] = 'x_corr_std'
    sim_pars['par_sweep_vals'] = [0.1,0.2,0.3]

    #sim_pars['par_sweep_key'] = 'x_corr_mean'
    #sim_pars['par_sweep_vals'] = [0.1,0.2,0.3]

    #sim_pars['par_sweep_key'] = 'ext_OU_sigma'
    #sim_pars['par_sweep_vals'] = [1.0,2.0,3.5,5.0,10.0]

    #sim_pars['par_sweep_key'] = 'eta1_base'
    #sim_pars['par_sweep_vals'] = list(eta_base*0.000015*np.array([1.0,2.0,3.0,4.0,5.0,6.0]))
    #sim_pars['par_sweep_vals'] = list(eta_base*0.00002*np.array([1.0]))

    #sim_pars['par_sweep_key'] = 'homogenous_base'
    #sim_pars['par_sweep_vals'] = list(np.array([1.0,2.0,5.0,10.0,25.0]))

    sim_pars['par_sweep_key'] = 'W_density'
    sim_pars['par_sweep_vals'] = list([0.5,0.8])

    #sim_pars['par_sweep_key'] = 'alpha_range'
    #sim_pars['par_sweep_vals'] = [150.0]

    #sim_pars['par_sweep_key'] = 'r_0_range'
    #sim_pars['par_sweep_vals'] = [1.0,5.0,10.0,15.0]

    sim_pars['par_sweep_key_2'] = 'ext_OU_sigma'
    #sim_pars['par_sweep_vals_2'] = [1.0,2.0,3.0,4.0]
    sim_pars['par_sweep_vals_2'] = [3.0,5.0]
    #sim_pars['par_sweep_vals_2'] = [6.0,7.0,8.0,9.0,10.0]

    #sim_pars['par_sweep_key_2'] = 'alpha_range'
    #sim_pars['par_sweep_vals_2'] = [1.0,2.5,5.0,10.0,25.0]
    #sim_pars['par_sweep_key_2'] = 'ext_OU_sigma'
    #sim_pars['par_sweep_vals_2'] = [0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]

    #sim_pars['par_sweep_key_2'] = 'W_input'
    #sim_pars['par_sweep_vals_2'] = [250,500.0,1000.0]

    sim_pars['sim_title'] = 'RF_net_N_250_target_p5_Wmax_2_alpha_uniform_homogbase_5_scaling_p5_eta_2_OU_5_sweep_density_pc_measure_struct_5_sims_OU_5'

    sim_pars['pc_measure_pars'] = {
        'ext_OU_sigma': sim_pars['ext_OU_sigma'],
        'present_random_bars': False
    }

    do_sweep_pars = {
        'distribute_exps' : False,
        'N_sims': 5,
        'N_cores': 5
        #'W_density': 0.8,
        #'ext_OU_sigma': 5.0
    }
    sim_pars.update(do_sweep_pars)


def run_iter(sim_pars,par_val):
    iter_pars = sim_pars.copy()
    iter_pars[iter_pars['par_sweep_key']] = par_val
    print iter_pars[sim_pars['par_sweep_key']]
    exp_results = run_RF_net.run_exp_pop_coupling(iter_pars,iter_pars['N_sims'])
    print exp_results['selectivities_results_corr']
    print exp_results['selectivities_results_uncorr']
    print exp_results['selectivities_results_anticorr']
    py_scripts_yann.save_pickle_safe('/mnt/DATA/ysweeney/data/topdown_learning/'+str(sim_pars['sim_title'])+'_'+str(iter_pars['par_sweep_key'])+'_'+str(par_val)+'.pkl',exp_results)
    return exp_results

#rc = ipp.Client()
#dview = rc[:]
#dview_results = dview.apply(run_RF_net.run_exp_group_selectivity,sim_pars['N_sims'],sim_pars['T'],sim_pars['N_filters'],sim_pars['group_corr_size'],sim_pars['group_corr_strength'],sim_pars['t_res'])
#while(not dview_results.ready()):
#    time.sleep(5)
#exp_results = dview_results.get()

#combined_exp_results = {}
#for key in exp_results[0]:
#    combined_exp_results[key] = []
#    for exp_res in exp_results:
#        combined_exp_results[key].append(exp_res[key])

#run_RF_net.plot_exp_group_selectivity(combined_exp_results,len(rc),sim_pars['T'],sim_pars['N_filters'],sim_pars['group_corr_size'])

#plt.show()
    #plt.savefig('/mnt/DATA/ysweeney/data/topdown_learning/plots/'+sim_title+'.png')

def run_iter_distributed(sim_pars,i):
    iter_pars = sim_pars.copy()
    exp_results = run_RF_net.run_exp_pop_coupling(iter_pars,iter_pars['N_sims'],250000)
    #print exp_results['selectivities_results_corr']
    #print exp_results['selectivities_results_uncorr']
    #print exp_results['selectivities_results_anticorr']
    #py_scripts_yann.save_pickle_safe('/mnt/DATA/ysweeney/data/topdown_learning/'+str(sim_pars['sim_title'])+'_'+str(i)+'.pkl',exp_results)
    #L.append(exp_results)

    exp_dir = mkdtemp(dir="./")
    # create a filename for storing some data
    data_file = os.path.join(exp_dir,str(sim_pars['sim_title'])+'_'+str(i)+'.pkl')
    # assume some random results
    with open(data_file, 'wb') as f:
            print("writing results")
            cPickle.dump(exp_results, f)
    # add the result as an artifact, note that the name here is important
    # as sacred otherwise will try to save to the oddly named tmp subdirectory created
    ex.add_artifact(data_file, name=os.path.basename(data_file))
    # at the very end of the run delete the temporary directory
    # sacred will have taken care of copying all the results files over to the run directoy
    shutil.rmtree(exp_dir)

    return exp_results

def func_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return run_iter(*a_b)

@ex.command
def debug_single_core(sim_pars):
    iter_pars = sim_pars.copy()
    iter_pars[sim_pars['par_sweep_key']] = sim_pars['par_sweep_vals'][0]
    print iter_pars[par_sweep_key]
    exp_results = run_RF_net.run_exp_pop_coupling(iter_pars,sim_pars['N_sims'])
    print exp_results['selectivities_results_corr']
    print exp_results['selectivities_results_uncorr']
    print exp_results['selectivities_results_anticorr']

    exp_dir = mkdtemp(dir="./")
    # create a filename for storing some data
    data_file = os.path.join(exp_dir,str(sim_pars['sim_title'])+'.pkl')
    # assume some random results
    with open(data_file, 'wb') as f:
            print("writing results")
            cPickle.dump(exp_results, f)
    # add the result as an artifact, note that the name here is important
    # as sacred otherwise will try to save to the oddly named tmp subdirectory created
    ex.add_artifact(data_file, name=os.path.basename(data_file))
    # at the very end of the run delete the temporary directory
    # sacred will have taken care of copying all the results files over to the run directoy
    shutil.rmtree(exp_dir)

    return exp_results


def consolidate_distrib_results(file_path,num_exps,plot_hist=True):
    results = []
    #for i in xrange(num_exps):
    #    results.append(py_scripts_yann.load_pickle(file_path+'_'+str(i)+'.pkl'))
    import glob
    file_paths = glob.glob(file_path+'*.pkl')
    num_exps = len(file_paths)
    for i in xrange(num_exps):
        results.append(py_scripts_yann.load_pickle(file_paths[i]))

#    print 'pop coupling partial pval, corr ' , np.mean([results[i]['selectivities_results_corr'][0]['pop_coupling_partial_pval'] for i in xrange(num_exps)])
#    print 'pop coupling partial pval, uncorr ' , np.mean([results[i]['selectivities_results_uncorr'][0]['pop_coupling_partial_pval'] for i in xrange(num_exps)])
#    print 'pop coupling partial pval, anticorr ' , np.mean([results[i]['selectivities_results_anticorr'][0]['pop_coupling_partial_pval'] for i in xrange(num_exps)])
#    print 'pop coupling partial rval, corr ' , np.mean([results[i]['selectivities_results_corr'][0]['pop_coupling_partial_rval'] for i in xrange(num_exps)])
#    print 'pop coupling partial rval, uncorr ' , np.mean([results[i]['selectivities_results_uncorr'][0]['pop_coupling_partial_rval'] for i in xrange(num_exps)])
#    print 'pop coupling partial rval, anticorr ' , np.mean([results[i]['selectivities_results_anticorr'][0]['pop_coupling_partial_rval'] for i in xrange(num_exps)])
#    print 'pop coupling rval, corr ' , np.mean([results[i]['selectivities_results_corr'][0]['pop_coupling_rval'] for i in xrange(num_exps)])
#    print 'pop coupling rval, uncorr ' , np.mean([results[i]['selectivities_results_uncorr'][0]['pop_coupling_rval'] for i in xrange(num_exps)])
#    print 'pop coupling rval, anticorr ' , np.mean([results[i]['selectivities_results_anticorr'][0]['pop_coupling_rval'] for i in xrange(num_exps)])
#    print 'rate rval, corr ' , np.mean([results[i]['selectivities_results_corr'][0]['rate_rval'] for i in xrange(num_exps)])
#    print 'rate rval, uncorr ' , np.mean([results[i]['selectivities_results_uncorr'][0]['rate_rval'] for i in xrange(num_exps)])
#    print 'rate rval, anticorr ' , np.mean([results[i]['selectivities_results_anticorr'][0]['rate_rval'] for i in xrange(num_exps)])
#    print 'rate pval, corr ' , np.mean([results[i]['selectivities_results_corr'][0]['rate_pval'] for i in xrange(num_exps)])
#    print 'rate pval, uncorr ' , np.mean([results[i]['selectivities_results_uncorr'][0]['rate_pval'] for i in xrange(num_exps)])
#    print 'rate pval, anticorr ' , np.mean([results[i]['selectivities_results_anticorr'][0]['rate_pval'] for i in xrange(num_exps)])
#
#
#    print 'standard devs'
#    print 'pop coupling partial pval, corr ' , np.std([results[i]['selectivities_results_corr'][0]['pop_coupling_partial_pval'] for i in xrange(num_exps)])
#    print 'pop coupling partial pval, uncorr ' , np.std([results[i]['selectivities_results_uncorr'][0]['pop_coupling_partial_pval'] for i in xrange(num_exps)])
#    print 'pop coupling partial pval, anticorr ' , np.std([results[i]['selectivities_results_anticorr'][0]['pop_coupling_partial_pval'] for i in xrange(num_exps)])
#    print 'pop coupling partial rval, corr ' , np.std([results[i]['selectivities_results_corr'][0]['pop_coupling_partial_rval'] for i in xrange(num_exps)])
#    print 'pop coupling partial rval, uncorr ' , np.std([results[i]['selectivities_results_uncorr'][0]['pop_coupling_partial_rval'] for i in xrange(num_exps)])
#    print 'pop coupling partial rval, anticorr ' , np.std([results[i]['selectivities_results_anticorr'][0]['pop_coupling_partial_rval'] for i in xrange(num_exps)])
#    print 'pop coupling rval, corr ' , np.std([results[i]['selectivities_results_corr'][0]['pop_coupling_rval'] for i in xrange(num_exps)])
#    print 'pop coupling rval, uncorr ' , np.std([results[i]['selectivities_results_uncorr'][0]['pop_coupling_rval'] for i in xrange(num_exps)])
#    print 'pop coupling rval, anticorr ' , np.std([results[i]['selectivities_results_anticorr'][0]['pop_coupling_rval'] for i in xrange(num_exps)])


    if plot_hist:
        plt.hist([[results[i]['selectivities_results_corr'][0]['pop_coupling_partial_rval'] for i in xrange(num_exps)],[results[i]['selectivities_results_uncorr'][0]['pop_coupling_partial_rval'] for i in xrange(num_exps)],[results[i]['selectivities_results_anticorr'][0]['pop_coupling_partial_rval'] for i in xrange(num_exps)]])
        plt.show()

    return results

def consolidate_sweep_results(file_path,plot_hist=True):
    results = []
    #for i in xrange(num_exps):
    #    results.append(py_scripts_yann.load_pickle(file_path+'_'+str(i)+'.pkl'))
    import json
    json_data=open(file_path+'/config.json').read()
    sim_pars = json.loads(json_data)['sim_pars']

    sweep_results = {
        'pc_rval_mean': np.zeros((len(sim_pars['par_sweep_vals']),len(sim_pars['par_sweep_vals_2']))),
        'pc_rval_std': np.zeros((len(sim_pars['par_sweep_vals']),len(sim_pars['par_sweep_vals_2']))),
        'uniform_connprob_mean':np.zeros((len(sim_pars['par_sweep_vals']),len(sim_pars['par_sweep_vals_2']))),
        'uniform_connprob_std':np.zeros((len(sim_pars['par_sweep_vals']),len(sim_pars['par_sweep_vals_2']))),
        'diverse_connprob_mean':np.zeros((len(sim_pars['par_sweep_vals']),len(sim_pars['par_sweep_vals_2']))),
        'diverse_connprob_std':np.zeros((len(sim_pars['par_sweep_vals']),len(sim_pars['par_sweep_vals_2']))),
        'diverse_pc_connprob':np.zeros((len(sim_pars['par_sweep_vals']),len(sim_pars['par_sweep_vals_2']))),
        'uniform_pc_connprob':np.zeros((len(sim_pars['par_sweep_vals']),len(sim_pars['par_sweep_vals_2']))),
        'diverse_pc_std':np.zeros((len(sim_pars['par_sweep_vals']),len(sim_pars['par_sweep_vals_2']))),
        'uniform_pc_std':np.zeros((len(sim_pars['par_sweep_vals']),len(sim_pars['par_sweep_vals_2']))),
        'diverse_pc_mean':np.zeros((len(sim_pars['par_sweep_vals']),len(sim_pars['par_sweep_vals_2']))),
        'uniform_pc_mean':np.zeros((len(sim_pars['par_sweep_vals']),len(sim_pars['par_sweep_vals_2']))),
        'diverse_selectivity_mean':np.zeros((len(sim_pars['par_sweep_vals']),len(sim_pars['par_sweep_vals_2']))),
        'uniform_selectivity_mean':np.zeros((len(sim_pars['par_sweep_vals']),len(sim_pars['par_sweep_vals_2']))),
        'diverse_selectivity_std':np.zeros((len(sim_pars['par_sweep_vals']),len(sim_pars['par_sweep_vals_2']))),
        'uniform_selectivity_std':np.zeros((len(sim_pars['par_sweep_vals']),len(sim_pars['par_sweep_vals_2']))),
        'diverse_selectivity_upper':np.zeros((len(sim_pars['par_sweep_vals']),len(sim_pars['par_sweep_vals_2']))),
        'uniform_selectivity_upper':np.zeros((len(sim_pars['par_sweep_vals']),len(sim_pars['par_sweep_vals_2']))),
        'uniform_selectivity_max':np.zeros((len(sim_pars['par_sweep_vals']),len(sim_pars['par_sweep_vals_2']))),
        'diverse_selectivity_max':np.zeros((len(sim_pars['par_sweep_vals']),len(sim_pars['par_sweep_vals_2']))),
        'uniform_pc_input':np.zeros((len(sim_pars['par_sweep_vals']),len(sim_pars['par_sweep_vals_2']))),
        'diverse_pc_input':np.zeros((len(sim_pars['par_sweep_vals']),len(sim_pars['par_sweep_vals_2']))),
    }

    for par_val_idx in xrange(len(sim_pars['par_sweep_vals'])):
        par_val = sim_pars['par_sweep_vals'][par_val_idx]
        if not sim_pars['par_sweep_key_2'] == None:
            iter_pars = sim_pars.copy()
            iter_pars[sim_pars['par_sweep_key']] = par_val
            for par_val_2_idx in xrange(len(sim_pars['par_sweep_vals_2'])):
                par_val_2 = sim_pars['par_sweep_vals_2'][par_val_2_idx]
                iter_pars[sim_pars['par_sweep_key_2']] = par_val_2
                str_i = sim_pars['par_sweep_key']+str(iter_pars[sim_pars['par_sweep_key']])+sim_pars['par_sweep_key_2']+str(iter_pars[sim_pars['par_sweep_key_2']])
                res_file = os.path.join(file_path,str(sim_pars['sim_title'])+'_'+str(str_i)+'.pkl')
                try:
                    temp_results = py_scripts_yann.load_pickle(res_file)
                    sweep_results['pc_rval_mean'][par_val_idx,par_val_2_idx] = np.mean([temp_results['selectivities_results_corr'][i]['pop_coupling_partial_rval'] for i in xrange(sim_pars['N_sims'])])
                    sweep_results['pc_rval_std'][par_val_idx,par_val_2_idx] = np.std([temp_results['selectivities_results_corr'][i]['pop_coupling_partial_rval'] for i in xrange(sim_pars['N_sims'])])

                    unif_connprob_temp = []
                    diverse_connprob_temp = []
                    #unif_connprob_std_temp = []
                    #diverse_connprob_std_temp = []
                    unif_corr_pc_connprob = []
                    diverse_corr_pc_connprob = []
                    unif_pc_temp= []
                    diverse_pc_temp = []
                    unif_input_temp= []
                    diverse_input_temp = []
                    uniform_selectivity_temp = []
                    diverse_selectivity_temp = []
                    unif_corr_pc_input = []
                    diverse_corr_pc_input = []
                    for i in xrange(sim_pars['N_sims']):
                        print 'sim idx ', i
                        unif_connprob_temp.append(np.mean(temp_results['simresults_uniform'][i]['W_conn'],axis=0))
                        diverse_connprob_temp.append(np.mean(temp_results['simresults_corr'][i]['W_conn'],axis=0))
                        unif_input_temp.append(np.sum(temp_results['simresults_uniform'][i]['W_plastic'],axis=0))
                        diverse_input_temp.append(np.sum(temp_results['simresults_corr'][i]['W_plastic'],axis=0))
                        #unif_connprob_std_temp.append(np.std(temp_results['simresults_uniform'][i]['W_conn'],axis=0))
                        #diverse_connprob_std_temp.append(np.std(temp_results['simresults_corr'][i]['W_conn'],axis=0))
                        unif_pc_temp.append(temp_results['selectivities_results_uniform'][i]['empirical_pop_coupling'])
                        diverse_pc_temp.append(temp_results['selectivities_results_corr'][i]['empirical_pop_coupling'])
                        unif_corr_pc_connprob.append(stats.pearsonr(temp_results['selectivities_results_uniform'][i]['empirical_pop_coupling'],unif_connprob_temp[-1]))
                        diverse_corr_pc_connprob.append(stats.pearsonr(temp_results['selectivities_results_corr'][i]['empirical_pop_coupling'],diverse_connprob_temp[-1]))
                        unif_corr_pc_input.append(stats.pearsonr(temp_results['selectivities_results_uniform'][i]['empirical_pop_coupling'],unif_input_temp[-1]))
                        diverse_corr_pc_input.append(stats.pearsonr(temp_results['selectivities_results_corr'][i]['empirical_pop_coupling'],diverse_input_temp[-1]))
                        uniform_selectivity_temp.append(temp_results['selectivities_t_uniform'][-1])
                        diverse_selectivity_temp.append(temp_results['selectivities_t_corr'][-1])
                    sweep_results['uniform_connprob_mean'][par_val_idx,par_val_2_idx] = np.mean(np.array(unif_connprob_temp))
                    sweep_results['uniform_connprob_std'][par_val_idx,par_val_2_idx] = np.std(np.array(unif_connprob_temp))
                    sweep_results['diverse_connprob_mean'][par_val_idx,par_val_2_idx] = np.mean(np.array(diverse_connprob_temp))
                    sweep_results['diverse_connprob_std'][par_val_idx,par_val_2_idx] = np.std(np.array(diverse_connprob_temp))
                    sweep_results['diverse_pc_connprob'][par_val_idx,par_val_2_idx] = np.mean(np.array(diverse_corr_pc_connprob))
                    sweep_results['uniform_pc_connprob'][par_val_idx,par_val_2_idx] = np.mean(np.array(unif_corr_pc_connprob))
                    sweep_results['diverse_pc_input'][par_val_idx,par_val_2_idx] = np.mean(np.array(diverse_corr_pc_input))
                    sweep_results['uniform_pc_input'][par_val_idx,par_val_2_idx] = np.mean(np.array(unif_corr_pc_input))
                    sweep_results['diverse_pc_mean'][par_val_idx,par_val_2_idx] = np.mean(np.array(diverse_pc_temp))
                    sweep_results['uniform_pc_mean'][par_val_idx,par_val_2_idx] = np.mean(np.array(unif_pc_temp))
                    sweep_results['diverse_pc_std'][par_val_idx,par_val_2_idx] = np.std(np.array(diverse_pc_temp))
                    sweep_results['uniform_pc_std'][par_val_idx,par_val_2_idx] = np.std(np.array(unif_pc_temp))
                    sweep_results['diverse_selectivity_mean'][par_val_idx,par_val_2_idx] = np.mean(np.array(diverse_selectivity_temp))
                    sweep_results['diverse_selectivity_std'][par_val_idx,par_val_2_idx] = np.std(np.array(diverse_selectivity_temp))
                    sweep_results['diverse_selectivity_upper'][par_val_idx,par_val_2_idx] = np.percentile(np.array(diverse_selectivity_temp),90)
                    sweep_results['diverse_selectivity_max'][par_val_idx,par_val_2_idx] = np.max(np.array(diverse_selectivity_temp))
                    sweep_results['uniform_selectivity_mean'][par_val_idx,par_val_2_idx] =np.mean(np.array(uniform_selectivity_temp))
                    sweep_results['diverse_selectivity_std'][par_val_idx,par_val_2_idx] = np.std(np.array(diverse_selectivity_temp))
                    sweep_results['uniform_selectivity_upper'][par_val_idx,par_val_2_idx] = np.percentile(np.array(uniform_selectivity_temp),90)
                    sweep_results['uniform_selectivity_max'][par_val_idx,par_val_2_idx] = np.max(np.array(uniform_selectivity_temp))
                    plt.figure()
                    plt.hist([np.array(unif_connprob_temp),np.array(diverse_connprob_temp)],20)
                    plt.legend(['uniform','diverse'])
                    plt.savefig(os.path.join(file_path,str(sim_pars['sim_title'])+'_'+str(str_i)+'_connprob_hist.pdf'))
                    plt.cla()

                    sns.jointplot(np.array(diverse_pc_temp),np.array(diverse_connprob_temp),kind='hexbin',ylim=(0.0,1.0))
                    plt.title('Plasticity-connectivity link, diverse')
                    plt.savefig(os.path.join(file_path,str(sim_pars['sim_title'])+'_'+str(str_i)+'pc_connprob_diverse.pdf'))
                    plt.cla()
                    sns.jointplot(np.array(unif_pc_temp),np.array(unif_connprob_temp),kind='hexbin',ylim=(0.0,1.0))
                    plt.title('Plasticity-connectivity link, uniform')
                    plt.savefig(os.path.join(file_path,str(sim_pars['sim_title'])+'_'+str(str_i)+'_pc_connprob_uniform.pdf'))
                    plt.cla()

                    plt.figure()
                    digit = np.digitize(np.array(diverse_pc_temp).flatten(),np.arange(-1.0,1.0,0.2))
                    plt.plot([np.mean(np.array(diverse_connprob_temp).flatten()[digit==i]) for i in xrange(10)])
                    digit = np.digitize(np.array(unif_pc_temp).flatten(),np.arange(-1.0,1.0,0.1))
                    plt.plot([np.mean(np.array(unif_connprob_temp).flatten()[digit==i]) for i in xrange(10)])
                    plt.title('PC-connectivity link')
                    plt.legend(['diverse','uniform'])
                    plt.savefig(os.path.join(file_path,str(sim_pars['sim_title'])+'_'+str(str_i)+'_pc_connprob_link.pdf'))
                    plt.cla()
                except:
                    pass
        else:
            iter_pars = sim_pars.copy()
            iter_pars[sim_pars['par_sweep_key']] = par_val
            str_i = sim_pars['par_sweep_key']+str(iter_pars[sim_pars['par_sweep_key']])  # Passing the list
            res_file = os.path.join(file_path,str(sim_pars['sim_title'])+'_'+str(str_i)+'.pkl')
            temp_results = py_scripts_yann.load_pickle(res_file)
            sweep_results['pc_rval_mean'][par_val_idx,par_val_2_idx] = np.mean([temp_results['selectivities_results_corr'][i]['pop_coupling_partial_rval'] for i in xrange(sim_pars['N_sims'])])
            sweep_results['pc_rval_std'][par_val_idx,par_val_2_idx] = np.std([temp_results['selectivities_results_corr'][i]['pop_coupling_partial_rval'] for i in xrange(sim_pars['N_sims'])])

    if plot_hist:
        plt.pcolor(sweep_results['pc_rval_mean'])
        plt.title('Plasticity-coupling link')
        plt.colorbar()
        plt.savefig(os.path.join(file_path,str(sim_pars['sim_title'])+'_'+str(str_i)+'_plasticity_coupling_link_mean.pdf'))
        plt.show()
        plt.pcolor(sweep_results['pc_rval_std'])
        plt.title('Plasticity-coupling link variability')
        plt.colorbar()
        plt.savefig(os.path.join(file_path,str(sim_pars['sim_title'])+'_'+str(str_i)+'_plasticity_coupling_link_std.pdf'))
        plt.show()
        plt.pcolor(sweep_results['uniform_connprob_std'])
        plt.title('input connectivity width, uniform')
        plt.colorbar()
        plt.savefig(os.path.join(file_path,str(sim_pars['sim_title'])+'_'+str(str_i)+'_connectivity_width_uniform.pdf'))
        plt.show()
        plt.pcolor(sweep_results['diverse_connprob_std'])
        plt.title('input connectivity width, diverse')
        plt.savefig(os.path.join(file_path,str(sim_pars['sim_title'])+'_'+str(str_i)+'_connectivity_width_diverse.pdf'))
        plt.colorbar()
        plt.show()
        plt.pcolor(sweep_results['diverse_pc_connprob'])
        plt.title('pc-connprob link, diverse')
        plt.colorbar()
        plt.savefig(os.path.join(file_path,str(sim_pars['sim_title'])+'_'+str(str_i)+'_pc_connprob_diverse.pdf'))
        plt.show()
        plt.pcolor(sweep_results['uniform_pc_connprob'])
        plt.title('pc-connprob link, uniform')
        plt.colorbar()
        plt.savefig(os.path.join(file_path,str(sim_pars['sim_title'])+'_'+str(str_i)+'_pc_connprob_uniform.pdf'))
        plt.show()
    return sweep_results

@ex.main
def auto_main(sim_pars):
    print 'running multicore'
    from multiprocessing import Pool, Process, Manager
    if sim_pars['distribute_exps']:
        with Manager() as manager:
            #exp_results_list = manager.list()  # <-- can be shared between processes.
            processes = []
            sim_pars['N_sims'] = int(sim_pars['N_sims']/sim_pars['N_cores'])
            for i in range(sim_pars['N_cores']):
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
            for par_val in sim_pars['par_sweep_vals']:
                if not sim_pars['par_sweep_key_2'] == None:
                    iter_pars = sim_pars.copy()
                    iter_pars[sim_pars['par_sweep_key']] = par_val
                    for par_val_2 in sim_pars['par_sweep_vals_2']:
                        iter_pars[sim_pars['par_sweep_key_2']] = par_val_2
                        iter_pars['W_max'] = sim_pars['W_max']*(1.0/iter_pars['W_density'])
                        #iter_pars['scaling_rate'] = sim_pars['scaling_rate']*(iter_pars['alpha_range'])
                        p = Process(target=run_iter_distributed, args=(iter_pars,sim_pars['par_sweep_key']+str(iter_pars[sim_pars['par_sweep_key']])+sim_pars['par_sweep_key_2']+str(iter_pars[sim_pars['par_sweep_key_2']])))  # Passing the list
                        np.random.seed()
                        p.start()
                        processes.append(p)
                else:
                    iter_pars = sim_pars.copy()
                    iter_pars[sim_pars['par_sweep_key']] = par_val
                    p = Process(target=run_iter_distributed, args=(iter_pars,sim_pars['par_sweep_key']+str(iter_pars[sim_pars['par_sweep_key']])))  # Passing the list
                    np.random.seed()
                    p.start()
                    processes.append(p)
            for p in processes:
                p.join()

        #p = Pool(len(sim_pars['par_sweep_vals']))
        #p.map(run_iter,par_sweep_vals)
        #p.map(func_star, itertools.izip(sim_pars['par_sweep_vals'], itertools.repeat(sim_pars)))


if __name__ == "__main__":
    ex.run()
