from allensdk.brain_observatory.stimulus_analysis import StimulusAnalysis
import scipy.stats as st
import pandas as pd
import numpy as np
from math import sqrt
import logging
from allensdk.brain_observatory.brain_observatory_exceptions import BrainObservatoryAnalysisException
from allensdk.brain_observatory.receptive_field_analysis.utilities import smooth


class DriftingGratings_yann(StimulusAnalysis):
    """ Perform tuning analysis specific to drifting gratings stimulus.

    Parameters
    ----------
    data_set: BrainObservatoryNwbDataSet object
    """

    _log = logging.getLogger('allensdk.brain_observatory.drifting_gratings')

    def __init__(self, data_set, **kwargs):
        super(DriftingGratings_yann, self).__init__(data_set, **kwargs)
        stimulus_table = self.data_set.get_stimulus_table('drifting_gratings')
        stimulus_table.fillna(value=0.)
        self.stim_table_all = stimulus_table
        self.sweeplength = 60
        self.interlength = 30
        self.extralength = 0
        self.orivals = np.unique(self.stim_table_all.orientation).astype(int)
        self.tfvals = np.unique(self.stim_table_all.temporal_frequency).astype(int)
        self.number_ori = len(self.orivals)
        self.number_tf = len(self.tfvals)
        self.sweep_response_all, self.mean_sweep_response_all, self.pval_all = self.get_sweep_response_all()

    def get_sweep_response_all(self):
        """ Calculates the response to each sweep in the stimulus table for each cell and the mean response.
        The return is a 3-tuple of:

            * sweep_response: pd.DataFrame of response dF/F traces organized by cell (column) and sweep (row)

            * mean_sweep_response: mean values of the traces returned in sweep_response

            * pval: p value from 1-way ANOVA comparing response during sweep to response prior to sweep

        Returns
        -------
        3-tuple: sweep_response, mean_sweep_response, pval
        """
        def do_mean(x):
            # +1])
            return np.mean(x[self.interlength:self.interlength + self.sweeplength + self.extralength])

        def do_p_value(x):
            (_, p) = st.f_oneway(x[:self.interlength], x[
                self.interlength:self.interlength + self.sweeplength + self.extralength])
            return p

        StimulusAnalysis._log.info('Calculating responses for each sweep')
        sweep_response = pd.DataFrame(index=self.stim_table_all.index.values, columns=np.array(
            range(self.numbercells + 1)).astype(str))
        sweep_response.rename(
            columns={str(self.numbercells): 'dx'}, inplace=True)
        for index, row in self.stim_table_all.iterrows():
            start = int(row['start'] - self.interlength)
            end = int(row['start'] + self.sweeplength + self.interlength)

            #print start, end, row

            for nc in range(self.numbercells):
                temp = self.celltraces[nc, start:end]
                sweep_response[str(nc)][index] = 100 * \
                    ((temp / np.mean(temp[:self.interlength])) - 1)
            sweep_response['dx'][index] = self.dxcm[start:end]

        mean_sweep_response = sweep_response.applymap(do_mean)

        pval = sweep_response.applymap(do_p_value)
        return sweep_response, mean_sweep_response, pval

    def get_response(self,start_time,end_time):
        ''' Computes the mean response for each cell to each stimulus condition.  Return is
        a (# orientations, # temporal frequencies, # cells, 3) np.ndarray.  The final dimension
        contains the mean response to the condition (index 0), standard error of the mean of the response
        to the condition (index 1), and the number of trials with a significant response (p < 0.05)
        to that condition (index 2).

        Returns
        -------
        Numpy array storing mean responses.
        '''
        DriftingGratings_yann._log.info("Calculating mean responses")

        print 'test', start_time, end_time
        self.time_slice = (np.logical_and([self.stim_table_all.start>start_time], [(self.stim_table_all.start+self.sweeplength+self.interlength)<end_time])[0])

        self.stim_table_t = self.stim_table_all[self.time_slice]

        #self.sweep_response, self.mean_sweep_response, self.pval = self.get_sweep_response_yann(start_time,end_time)

        self.sweep_response_t = self.sweep_response_all[self.time_slice]
        self.mean_sweep_response_t = self.mean_sweep_response_all[self.time_slice]
        self.pval_t = self.pval_all[self.time_slice]

        response = np.empty(
            (self.number_ori, self.number_tf, self.numbercells + 1, 3))

        def ptest(x):
            return len(np.where(x < (0.05 / (8 * 5)))[0])

        for ori in self.orivals:
            ori_pt = np.where(self.orivals == ori)[0][0]
            for tf in self.tfvals:
                tf_pt = np.where(self.tfvals == tf)[0][0]
                subset_response = self.mean_sweep_response_t[
                    (self.stim_table_t.temporal_frequency == tf) & (self.stim_table_t.orientation == ori)]
                subset_pval = self.pval_t[(self.stim_table_t.temporal_frequency == tf) & (
                    self.stim_table_t.orientation == ori)]
                response[ori_pt, tf_pt, :, 0] = subset_response.mean(axis=0)
                response[ori_pt, tf_pt, :, 1] = subset_response.std(
                    axis=0) / sqrt(len(subset_response))
                response[ori_pt, tf_pt, :, 2] = subset_pval.apply(
                    ptest, axis=0)

        self.response_t = response
        return response

    def get_peak(self,start_time,end_time):
        ''' Computes metrics related to each cell's peak response condition.

        Returns
        -------
        Pandas data frame containing the following columns (_dg suffix is
        for drifting grating):
            * ori_dg (orientation)
            * tf_dg (temporal frequency)
            * response_reliability_dg
            * osi_dg (orientation selectivity index)
            * dsi_dg (direction selectivity index)
            * peak_dff_dg (peak dF/F)
            * ptest_dg
            * p_run_dg
            * run_modulation_dg
            * cv_dg (circular variance)
        '''
        DriftingGratings_yann._log.info('Calculating peak response properties')

        peak_t = pd.DataFrame(index=range(self.numbercells), columns=('ori_dg', 'tf_dg', 'response_reliability_dg',
                                                                    'osi_dg', 'dsi_dg', 'peak_dff_dg', 'ptest_dg', 'p_run_dg', 'run_modulation_dg', 'cv_dg', 'cell_specimen_id'))
        cids = self.data_set.get_cell_specimen_ids()

        orivals_rad = np.deg2rad(self.orivals)

        self.time_slice = (np.logical_and([self.stim_table_all.start>start_time], [(self.stim_table_all.start+self.sweeplength+self.interlength)<end_time])[0])

        self.stim_table_t = self.stim_table_all[self.time_slice]

        self.sweep_response_t = self.sweep_response_all[self.time_slice]
        self.mean_sweep_response_t = self.mean_sweep_response_all[self.time_slice]
        self.pval_t = self.pval_all[self.time_slice]

        self.response_t = self.get_response(start_time,end_time)

        for nc in range(self.numbercells):
            cell_peak = np.where(self.response_t[:, 1:, nc, 0] == np.nanmax(
                self.response_t[:, 1:, nc, 0]))
            prefori = cell_peak[0][0]
            preftf = cell_peak[1][0] + 1
            peak_t.cell_specimen_id.iloc[nc] = cids[nc]
            peak_t.ori_dg.iloc[nc] = prefori
            peak_t.tf_dg.iloc[nc] = preftf
            peak_t.response_reliability_dg.iloc[
                nc] = self.response_t[prefori, preftf, nc, 2] / 0.15
            pref = self.response_t[prefori, preftf, nc, 0]
            orth1 = self.response_t[np.mod(prefori + 2, 8), preftf, nc, 0]
            orth2 = self.response_t[np.mod(prefori - 2, 8), preftf, nc, 0]
            orth = (orth1 + orth2) / 2
            null = self.response_t[np.mod(prefori + 4, 8), preftf, nc, 0]

            tuning = self.response_t[:, preftf, nc, 0]
            CV_top = np.empty((8))
            for i in range(8):
                CV_top[i] = (tuning[i] * np.exp(1j * 2 * orivals_rad[i])).real
            peak_t.cv_dg.iloc[nc] = np.abs(CV_top.sum() / tuning.sum())

            peak_t.osi_dg.iloc[nc] = (pref - orth) / (pref + orth)
            peak_t.dsi_dg.iloc[nc] = (pref - null) / (pref + null)
            peak_t.peak_dff_dg.iloc[nc] = pref

            groups = []
            for ori in self.orivals:
                for tf in self.tfvals[1:]:
                    groups.append(self.mean_sweep_response_t[(self.stim_table_t.temporal_frequency == tf) & (
                        self.stim_table_t.orientation == ori)][str(nc)])
            groups.append(self.mean_sweep_response_t[
                          self.stim_table_t.temporal_frequency == 0][str(nc)])
            f, p = st.f_oneway(*groups)
            peak_t.ptest_dg.iloc[nc] = p

            subset = self.mean_sweep_response_t[(self.stim_table_t.temporal_frequency == self.tfvals[
                                               preftf]) & (self.stim_table_t.orientation == self.orivals[prefori])]
            subset_stat = subset[subset.dx < 1]
            subset_run = subset[subset.dx >= 1]
            if (len(subset_run) > 2) & (len(subset_stat) > 2):
                (f, peak_t.p_run_dg.iloc[nc]) = st.ks_2samp(
                    subset_run[str(nc)], subset_stat[str(nc)])
                peak_t.run_modulation_dg.iloc[nc] = subset_run[
                    str(nc)].mean() / subset_stat[str(nc)].mean()
            else:
                peak_t.p_run_dg.iloc[nc] = np.NaN
                peak_t.run_modulation_dg.iloc[nc] = np.NaN


        self.peak_t = peak_t
        return peak_t

    def detect_events(self,  start_time, end_time, debug_plots=False):
        self.time_slice = (np.logical_and([self.stim_table_all.start>start_time], [(self.stim_table_all.start+self.sweeplength+self.interlength)<end_time])[0])

        stimulus_table = self.stim_table_all[self.time_slice]
        cids = self.data_set.get_cell_specimen_ids()
        b = np.zeros((len(cids),len(stimulus_table)), dtype=np.bool)

        k_min = 0
        k_max = 10
        delta = 3

        var_dict = {}
        debug_dict = {}
        for nc in range(self.numbercells):
            dff_trace = self.data_set.get_dff_traces()[1][nc, :]
            dff_trace = smooth(dff_trace, 5)
            for ii, fi in enumerate(stimulus_table['start'].values):

                if ii > 0 and stimulus_table.iloc[ii].start == stimulus_table.iloc[ii-1].end:
                    offset = 1
                else:
                    offset = 0

                if fi + k_min >= 0 and fi + k_max <= len(dff_trace):
                    trace = dff_trace[fi + k_min+1+offset:fi + k_max+1+offset]

                    xx = (trace - trace[0])[delta] - (trace - trace[0])[0]
                    yy = max((trace - trace[0])[delta + 2] - (trace - trace[0])[0 + 2],
                             (trace - trace[0])[delta + 3] - (trace - trace[0])[0 + 3],
                             (trace - trace[0])[delta + 4] - (trace - trace[0])[0 + 4])

                    var_dict[ii] = (trace[0], trace[-1], xx, yy)
                    debug_dict[fi + k_min+1+offset] = (ii, trace)

            xx_list, yy_list = [], []
            for _, _, xx, yy in var_dict.values():
                xx_list.append(xx)
                yy_list.append(yy)

            mu_x = np.median(xx_list)
            mu_y = np.median(yy_list)

            xx_centered = np.array(xx_list)-mu_x
            yy_centered = np.array(yy_list)-mu_y

            std_factor = 1
            std_x = 1./std_factor*np.percentile(np.abs(xx_centered), [100*(1-2*(1-st.norm.cdf(std_factor)))])
            std_y = 1./std_factor*np.percentile(np.abs(yy_centered), [100*(1-2*(1-st.norm.cdf(std_factor)))])

            curr_inds = []
            allowed_sigma = 4
            for ii, (xi, yi) in enumerate(zip(xx_centered, yy_centered)):
                if np.sqrt(((xi)/std_x)**2+((yi)/std_y)**2) < allowed_sigma:
                    curr_inds.append(True)
                else:
                    curr_inds.append(False)

            curr_inds = np.array(curr_inds)
            data_x = xx_centered[curr_inds]
            data_y = yy_centered[curr_inds]
            Cov = np.cov(data_x, data_y)
            Cov_Factor = np.linalg.cholesky(Cov)
            Cov_Factor_Inv = np.linalg.inv(Cov_Factor)

            #===================================================================================================================

            noise_threshold = max(allowed_sigma * std_x + mu_x, allowed_sigma * std_y + mu_y)
            mu_array = np.array([mu_x, mu_y])
            yes_set, no_set = set(), set()
            for ii, (t0, tf, xx, yy) in var_dict.items():


                xi_z, yi_z = Cov_Factor_Inv.dot((np.array([xx,yy]) - mu_array))

                # Conditions in order:
                # 1) Outside noise blob
                # 2) Minimum change in df/f
                # 3) Change evoked by this trial, not previous
                # 4) At end of trace, ended up outside of noise floor

                if np.sqrt(xi_z**2 + yi_z**2) > 4 and yy > .05 and xx < yy and tf > noise_threshold/2:
                    yes_set.add(ii)
                else:
                    no_set.add(ii)



            assert len(var_dict) == len(stimulus_table)
            for yi in yes_set:
                b[nc,yi] = True

            if debug_plots == True:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1,2)
                # ax[0].plot(dff_trace)
                for key, val in debug_dict.items():
                    ti, trace = val
                    if ti in no_set:
                        ax[0].plot(np.arange(key, key+len(trace)), trace, 'b')
                    elif ti in yes_set:
                        ax[0].plot(np.arange(key, key + len(trace)), trace, 'r', linewidth=2)
                    else:
                        raise Exception

                for ii in yes_set:
                    ax[1].plot([var_dict[ii][2]], [var_dict[ii][3]], 'r.')

                for ii in no_set:
                    ax[1].plot([var_dict[ii][2]], [var_dict[ii][3]], 'b.')

                print('number_of_events: %d' % b.sum())
                plt.show()

        return b

