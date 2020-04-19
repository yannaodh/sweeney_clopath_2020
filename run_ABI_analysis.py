import numpy as np
import py_scripts_yann
import scipy
import seaborn as sns
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from matplotlib import pyplot as plt
from scipy import stats

import DriftingGratings_yann
import StaticGratings_yann

# This class uses a 'manifest' to keep track of downloaded data and metadata.
# All downloaded files will be stored relative to the directory holding the manifest
# file.  If 'manifest_file' is a relative path (as it is below), it will be
# saved relative to your working directory.  It can also be an absolute path.
boc = BrainObservatoryCache(manifest_file='/mnt/DATA/ysweeney/data/topdown_learning/ABI_data/boc/manifest.json')


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
    #print 'partial correlation', corr

    if plot:
        fig,axes=plt.subplots()
        plt.scatter(res_j,res_i,color=color)
        plt.title('Residuals ,' + title + ' ; ' + str(corr[0])+ ' ' + str(corr[1]))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

def plot_pop_coupling_link(exp_id,foo):
    data_set = boc.get_ophys_experiment_data(exp_id)
    print data_set.list_stimuli()
    if 'drifting_gratings' not in data_set.list_stimuli() and 'static_gratings' not in data_set.list_stimuli():
        print 'no static or drifting_gratings'

        return None, None

    print exp_id, len(data_set.get_cell_specimen_ids())

    time, dff_traces = data_set.get_corrected_fluorescence_traces(cell_specimen_ids=data_set.get_cell_specimen_ids()[:])
    print dff_traces.shape[0]

    dff_means = np.mean(dff_traces,axis=1)
    pop_avg = np.mean(dff_traces,axis=0)

    dff_norms = dff_traces.copy()
    for i in xrange(dff_means.shape[0]):
        dff_norms[i,:] = dff_traces[i,:]/dff_means[i]

    dff_non_normed = dff_traces.copy()
    dff_traces = dff_norms.copy()

    corr_coeff = np.corrcoef(dff_traces)

    pop_corr = np.zeros(corr_coeff.shape[0])
    #np.random.shuffle(pop_avg)

    popcorr_1st_half = np.zeros(corr_coeff.shape[0])
    popcorr_2nd_half = np.zeros(corr_coeff.shape[0])

    N_pop = dff_traces.shape[0]
    #print scipy.stats.pearsonr(np.mean(dff_traces[:int(N_pop*0.5),:],axis=0),np.mean(dff_traces[int(N_pop*0.5):,:],axis=0))

    for i in xrange(corr_coeff.shape[0]):
        pop_corr[i] = np.corrcoef(dff_traces[i,:],pop_avg)[0][1]
        #popcorr_1st_half[i] = np.corrcoef(dff_traces[i,:(dff_traces.shape[1]*0.5)],pop_avg[:(dff_traces.shape[1]*0.5)])[0][1]
        #popcorr_2nd_half[i] = np.corrcoef(dff_traces[i,table['start'][0]:table['end'][0]],pop_avg[table['start'][0]:table['end'][0]])[0][1]
        #popcorr_2nd_half[i] = np.corrcoef(dff_traces[i,(40000+table['start'][0]):(40000+table['end'][0])],pop_avg[(40000+table['start'][0]):(40000+table['end'][0])])[0][1]


    Nbins = 12
    popcorr_bins = np.zeros((corr_coeff.shape[0],Nbins))
    binlength = int(dff_traces.shape[1]/Nbins)


    poprates_bins = np.zeros((corr_coeff.shape[0],Nbins))

    for j in xrange(Nbins):
        for i in xrange(corr_coeff.shape[0]):
            mask = np.zeros(corr_coeff.shape[0])
            mask[i] = 1
            masked_arr = np.ma.masked_array(dff_traces)

            popcorr_bins[i,j] = np.corrcoef(dff_traces[i,j*binlength:(j+1)*binlength],pop_avg[j*binlength:(j+1)*binlength])[0][1]
            poprates_bins[i,j] = np.mean(dff_non_normed[i,j*binlength:(j+1)*binlength])
    #plt.hist(pop_corr)

    #plt.figure()
    #plt.scatter(popcorr_bins[:,0],popcorr_bins[:,-1])
    #plt.scatter(popcorr_1st_half,np.abs((popcorr_2nd_half-popcorr_1st_half)),color='r')
    #plt.scatter(popcorr_1st_half,np.abs((popcorr_2nd_half)),color='g')


    #plt.figure()
    #plt.plot(np.transpose(popcorr_bins[:50,:]),alpha=0.4)
    #plt.plot(np.mean(popcorr_bins,axis=0),lw=3)

    #plt.figure()
    #plt.scatter(np.mean(popcorr_bins,axis=1),np.std(popcorr_bins,axis=1))


    #print scipy.stats.pearsonr(np.mean(popcorr_bins,axis=1),np.std(popcorr_bins,axis=1))
    #print 'mean popcorr = ', np.mean(popcorr_bins)
    partial_corr(np.std(popcorr_bins,axis=1),np.mean(popcorr_bins,axis=1),np.mean(poprates_bins,axis=1),plot=False)
    partial_corr(np.std(popcorr_bins,axis=1),np.mean(popcorr_bins,axis=1),np.std(poprates_bins,axis=1),plot=False)

    py_scripts_yann.save_pickle('/mnt/DATA/ysweeney/data/topdown_learning/ABI_data/analysis/'+str(exp_id)+'_rates_bins.pkl',poprates_bins)

    if scipy.stats.pearsonr(np.mean(dff_traces[:int(N_pop*0.5),:],axis=0),np.mean(dff_traces[int(N_pop*0.5):,:],axis=0))[0] < 0.8:
        print 'failed ', data_set.list_stimuli()
        return None, None
    if 'drifting_gratings' in data_set.list_stimuli():
        print 'DRIFT'
        dg = DriftingGratings_yann.DriftingGratings_yann(data_set)

        dg_early_peak = dg.get_peak(0 ,80000)
        #dg_early_peak_1 = dg.get_peak(0 ,40000)
        #dg_early_peak_2 = dg.get_peak(40000 ,80000)
        dg_early_response = dg.response_t
        dg_early_events = dg.detect_events(0 ,80000)
        # filter for visually responding, selective cells
        vis_cells_early = (dg.peak_t.ptest_dg < 0.05) &  (dg.peak_t.peak_dff_dg > 3)
        osi_cells_early = vis_cells_early & (dg.peak_t.osi_dg > 0.5) & (dg.peak_t.osi_dg <= 1.5)
        dsi_cells_early = vis_cells_early & (dg.peak_t.dsi_dg > 0.5) & (dg.peak_t.dsi_dg <= 1.5)

        dg_late_peak = dg.get_peak(80000 ,120000)
        dg_late_response = dg.response_t
        dg_late_events = dg.detect_events(80000 ,120000)

        # filter for visually responding, selective cells
        vis_cells_late = (dg.peak_t.ptest_dg < 0.05) &  (dg.peak_t.peak_dff_dg > 3)
        osi_cells_late = vis_cells_late & (dg.peak_t.osi_dg > 0.5) & (dg.peak_t.osi_dg <= 1.5)
        dsi_cells_late = vis_cells_late & (dg.peak_t.dsi_dg > 0.5) & (dg.peak_t.dsi_dg <= 1.5)

        early_set = set(np.where(vis_cells_early>0)[0])
        late_set = set(np.where(vis_cells_late>0)[0])

        preserved_coupling_result_sample = {}

        preserved_coupling_result_sample['vis'] = (vis_cells_early,vis_cells_late)
        preserved_coupling_result_sample['osi']=(osi_cells_early,osi_cells_late)
        preserved_coupling_result_sample['dsi']=(dsi_cells_early,dsi_cells_late)
        preserved_coupling_result_sample['cv_dg']=(dg_early_peak.cv_dg ,dg_late_peak.cv_dg )
        preserved_coupling_result_sample['tf_dg']=(dg_early_peak.tf_dg ,dg_late_peak.tf_dg )
        preserved_coupling_result_sample['ptest_dg']=(dg_early_peak.ptest_dg ,dg_late_peak.ptest_dg )
        preserved_coupling_result_sample['peak_dff_dg']=(dg_early_peak.peak_dff_dg ,dg_late_peak.peak_dff_dg )
        preserved_coupling_result_sample['osi_dg']=(dg_early_peak.osi_dg ,dg_late_peak.osi_dg )
        preserved_coupling_result_sample['dsi_dg']=(dg_early_peak.dsi_dg ,dg_late_peak.dsi_dg )
        preserved_coupling_result_sample['ori_dg']=(dg_early_peak.ori_dg ,dg_late_peak.ori_dg )
        preserved_coupling_result_sample['response_reliability_dg']=(dg_early_peak.response_reliability_dg ,dg_late_peak.response_reliability_dg )
        preserved_coupling_result_sample['responses_dg']=(dg_early_response,dg_late_response)

        preserved_mean_coupling_sample =  np.mean(np.mean(popcorr_bins,axis=1)[list(early_set.intersection(late_set))])
        non_preserved_mean_coupling_sample = np.mean(np.mean(popcorr_bins,axis=1)[list(early_set.difference(late_set).union(late_set.difference(early_set)))])

        print 'presevered mean pop coupling', np.mean(np.mean(popcorr_bins,axis=1)[list(early_set.intersection(late_set))])
        print 'non-presevered mean pop coupling', np.mean(np.mean(popcorr_bins,axis=1)[list(early_set.difference(late_set).union(late_set.difference(early_set)))])
        print scipy.stats.ttest_ind(np.mean(popcorr_bins,axis=1)[list(early_set.intersection(late_set))],np.mean(popcorr_bins,axis=1)[list(early_set.difference(late_set).union(late_set.difference(early_set)))])

    if 'static_gratings' in data_set.list_stimuli():
        print 'STAT'
        sg = StaticGratings_yann.StaticGratings_yann(data_set)

        sg_early_peak = sg.get_peak(0 ,20000)
        sg_early_response = sg.response_t
        sg_early_events = sg.detect_events(0 ,20000)
        # filter for visually responding, selective cells
        vis_cells_early = (sg.peak_t.ptest_sg < 0.05) &  (sg.peak_t.peak_dff_sg > 3)
        osi_cells_early = vis_cells_early & (sg.peak_t.osi_sg > 0.5) & (sg.peak_t.osi_sg <= 1.5)

        sg_late_peak = sg.get_peak(80000 ,120000)
        sg_late_response = sg.response_t
        sg_late_events = sg.detect_events(80000 ,120000)

        # filter for visually responding, selective cells
        vis_cells_late = (sg.peak_t.ptest_sg < 0.05) &  (sg.peak_t.peak_dff_sg > 3)
        osi_cells_late = vis_cells_late & (sg.peak_t.osi_sg > 0.5) & (sg.peak_t.osi_sg <= 1.5)

        early_set = set(np.where(vis_cells_early>0)[0])
        late_set = set(np.where(vis_cells_late>0)[0])

        preserved_coupling_result_sample = {}

        preserved_coupling_result_sample['vis'] = (vis_cells_early,vis_cells_late)
        preserved_coupling_result_sample['osi']=(osi_cells_early,osi_cells_late)
        preserved_coupling_result_sample['sf_sg']=(sg_early_peak.sf_sg ,sg_late_peak.sf_sg )
        preserved_coupling_result_sample['phase_sg']=(sg_early_peak.phase_sg ,sg_late_peak.phase_sg )
        preserved_coupling_result_sample['ptest_sg']=(sg_early_peak.ptest_sg ,sg_late_peak.ptest_sg )
        preserved_coupling_result_sample['peak_dff_sg']=(sg_early_peak.peak_dff_sg ,sg_late_peak.peak_dff_sg )
        preserved_coupling_result_sample['osi_sg']=(sg_early_peak.osi_sg ,sg_late_peak.osi_sg )
        preserved_coupling_result_sample['ori_sg']=(sg_early_peak.ori_sg ,sg_late_peak.ori_sg )
        preserved_coupling_result_sample['response_reliability_sg']=(sg_early_peak.response_reliability_sg,sg_late_peak.response_reliability_sg )
        preserved_coupling_result_sample['responses_sg']=(sg_early_response,sg_late_response)

        preserved_mean_coupling_sample =  np.mean(np.mean(popcorr_bins,axis=1)[list(early_set.intersection(late_set))])
        non_preserved_mean_coupling_sample = np.mean(np.mean(popcorr_bins,axis=1)[list(early_set.difference(late_set).union(late_set.difference(early_set)))])

        print 'presevered mean pop coupling', np.mean(np.mean(popcorr_bins,axis=1)[list(early_set.intersection(late_set))])
        print 'non-presevered mean pop coupling', np.mean(np.mean(popcorr_bins,axis=1)[list(early_set.difference(late_set).union(late_set.difference(early_set)))])
        print scipy.stats.ttest_ind(np.mean(popcorr_bins,axis=1)[list(early_set.intersection(late_set))],np.mean(popcorr_bins,axis=1)[list(early_set.difference(late_set).union(late_set.difference(early_set)))])


    py_scripts_yann.save_pickle('/mnt/DATA/ysweeney/data/topdown_learning/ABI_data/analysis/'+str(exp_id)+'_preserved_pc.pkl',preserved_coupling_result_sample)
    py_scripts_yann.save_pickle('/mnt/DATA/ysweeney/data/topdown_learning/ABI_data/analysis/'+str(exp_id)+'_pc_bins.pkl',popcorr_bins)
    py_scripts_yann.save_pickle('/mnt/DATA/ysweeney/data/topdown_learning/ABI_data/analysis/'+str(exp_id)+'_rates_bins.pkl',poprates_bins)
    if 'static_gratings' in data_set.list_stimuli():
        py_scripts_yann.save_pickle('/mnt/DATA/ysweeney/data/topdown_learning/ABI_data/analysis/'+str(exp_id)+'_early_events.pkl',sg_early_events)
        py_scripts_yann.save_pickle('/mnt/DATA/ysweeney/data/topdown_learning/ABI_data/analysis/'+str(exp_id)+'_late_events.pkl',sg_late_events)
    elif 'drifting_gratings' in data_set.list_stimuli():
        py_scripts_yann.save_pickle('/mnt/DATA/ysweeney/data/topdown_learning/ABI_data/analysis/'+str(exp_id)+'_early_events.pkl',dg_early_events)
        py_scripts_yann.save_pickle('/mnt/DATA/ysweeney/data/topdown_learning/ABI_data/analysis/'+str(exp_id)+'_late_events.pkl',dg_late_events)

if __name__ == '__main__':
    preserved_coupling_results = []
    popcorr_bins_results = []
    popcorr_mean_v_std_stats = []

    import os
    data_files = os.listdir('/mnt/DATA/ysweeney/data/topdown_learning/ABI_data/boc/ophys_experiment_data/')
    data_sets = []
    for data_file in data_files:
        data_sets.append(int(data_file.partition('.')[0]))

    color_pallete = sns.color_palette('deep',len(data_sets))

    print data_sets

    from multiprocessing import Process, Manager
    with Manager() as manager:
        #exp_results_list = manager.list()  # <-- can be shared between processes.
        processes = []
        for data_set_idx in xrange(len(data_sets)):
            p = Process(target=plot_pop_coupling_link, args=(data_sets[data_set_idx],None))  # Passing the list
            np.random.seed()
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

