# Copyright 2016 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.


import scipy.stats as st
import numpy as np
import pandas as pd
from math import sqrt
import logging
from allensdk.brain_observatory.stimulus_analysis import StimulusAnalysis
from allensdk.brain_observatory.brain_observatory_exceptions import BrainObservatoryAnalysisException
from allensdk.brain_observatory.receptive_field_analysis.utilities import smooth

class StaticGratings_yann(StimulusAnalysis):
    """ Perform tuning analysis specific to static gratings stimulus.

    Parameters
    ----------
    data_set: BrainObservatoryNwbDataSet object
    """

    _log = logging.getLogger('allensdk.brain_observatory.static_gratings')

    def __init__(self, data_set, **kwargs):
        super(StaticGratings_yann, self).__init__(data_set, **kwargs)

        stimulus_table = self.data_set.get_stimulus_table('static_gratings')
        stimulus_table.fillna(value=0.)
        self.stim_table_all = stimulus_table
        self.sweeplength = int(self.stim_table_all['end'].iloc[
            1] - self.stim_table_all['start'].iloc[1])
        self.interlength = int(4 * self.sweeplength)
        self.extralength = int(self.sweeplength)
        self.orivals = np.unique(self.stim_table_all.orientation.dropna())
        self.sfvals = np.unique(self.stim_table_all.spatial_frequency.dropna())
        self.phasevals = np.unique(self.stim_table_all.phase.dropna())
        self.number_ori = len(self.orivals)
        self.number_sf = len(self.sfvals)
        self.number_phase = len(self.phasevals)
        self.sweep_response_all, self.mean_sweep_response_all, self.pval_all = self.get_sweep_response_all()
        #self.response = self.get_response()
        #self.peak = self.get_peak()



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
        sweep_response = pd.DataFrame(index=self.stim_table_all.index.values,
                                      columns=map(str, range(self.numbercells + 1)))
        sweep_response.rename(
            columns={str(self.numbercells): 'dx'}, inplace=True)
        for index, row in self.stim_table_all.iterrows():
            start = int(row['start'] - self.interlength)
            end = int(row['start'] + self.sweeplength + self.interlength)

            for nc in range(self.numbercells):
                temp = self.celltraces[int(nc), start:end]
                sweep_response[str(nc)][index] = 100 * \
                    ((temp / np.mean(temp[:self.interlength])) - 1)
            sweep_response['dx'][index] = self.dxcm[start:end]

        mean_sweep_response = sweep_response.applymap(do_mean)

        pval = sweep_response.applymap(do_p_value)
        return sweep_response, mean_sweep_response, pval


    def get_response(self,start_time,end_time):
        ''' Computes the mean response for each cell to each stimulus condition.  Return is
        a (# orientations, # spatial frequencies, # phasees, # cells, 3) np.ndarray.  The final dimension
        contains the mean response to the condition (index 0), standard error of the mean of the response
        to the condition (index 1), and the number of trials with a significant response (p < 0.05)
        to that condition (index 2).

        Returns
        -------
        Numpy array storing mean responses.
        '''
        StaticGratings_yann._log.info("Calculating mean responses")

        self.time_slice = (np.logical_and([self.stim_table_all.start>start_time], [(self.stim_table_all.start+self.sweeplength+self.interlength)<end_time])[0])

        self.stim_table_t = self.stim_table_all[self.time_slice]

        #self.sweep_response, self.mean_sweep_response, self.pval = self.get_sweep_response_yann(start_time,end_time)
        #self.sweep_response_t, self.mean_sweep_response_t, self.pval_t = self.get_sweep_response_all()

        self.sweep_response_t = self.sweep_response_all[self.time_slice]
        self.mean_sweep_response_t = self.mean_sweep_response_all[self.time_slice]
        self.pval_t = self.pval_all[self.time_slice]

        response = np.empty((self.number_ori, self.number_sf,
                             self.number_phase, self.numbercells + 1, 3))

        def ptest(x):
            return len(np.where(x < (0.05 / (self.number_ori * (self.number_sf - 1))))[0])

        for ori in self.orivals:
            ori_pt = np.where(self.orivals == ori)[0][0]

            for sf in self.sfvals:
                sf_pt = np.where(self.sfvals == sf)[0][0]

                for phase in self.phasevals:
                    phase_pt = np.where(self.phasevals == phase)[0][0]
                    subset_response = self.mean_sweep_response_t[(self.stim_table_t.spatial_frequency == sf) & (
                        self.stim_table_t.orientation == ori) & (self.stim_table_t.phase == phase)]
                    subset_pval = self.pval_t[(self.stim_table_t.spatial_frequency == sf) & (
                        self.stim_table_t.orientation == ori) & (self.stim_table_t.phase == phase)]
                    response[ori_pt, sf_pt, phase_pt, :,
                             0] = subset_response.mean(axis=0)
                    response[ori_pt, sf_pt, phase_pt, :, 1] = subset_response.std(
                        axis=0) / sqrt(len(subset_response))
                    response[ori_pt, sf_pt, phase_pt, :,
                             2] = subset_pval.apply(ptest, axis=0)

        self.response_t = response
        return response

    def get_peak(self,start_time,end_time):
        ''' Computes metrics related to each cell's peak response condition.

        Returns
        -------
        Panda data frame with the following fields (_sg suffix is
        for static grating):
            * ori_sg (orientation)
            * sf_sg (spatial frequency)
            * phase_sg
            * response_variability_sg
            * osi_sg (orientation selectivity index)
            * peak_dff_sg (peak dF/F)
            * ptest_sg
            * time_to_peak_sg
            * duration_sg
        '''
        StaticGratings_yann._log.info('Calculating peak response properties')

        peak_t = pd.DataFrame(index=range(self.numbercells), columns=('ori_sg', 'sf_sg', 'phase_sg', 'response_reliability_sg',
                                                                    'osi_sg', 'peak_dff_sg', 'ptest_sg', 'time_to_peak_sg', 'duration_sg', 'cell_specimen_id'))
        cids = self.data_set.get_cell_specimen_ids()

        self.time_slice = (np.logical_and([self.stim_table_all.start>start_time], [(self.stim_table_all.start+self.sweeplength+self.interlength)<end_time])[0])

        self.stim_table_t = self.stim_table_all[self.time_slice]

        self.sweep_response_t = self.sweep_response_all[self.time_slice]
        self.mean_sweep_response_t = self.mean_sweep_response_all[self.time_slice]
        self.pval_t = self.pval_all[self.time_slice]

        self.response_t = self.get_response(start_time,end_time)

        for nc in range(self.numbercells):
            cell_peak = np.where(self.response_t[:, 1:, :, nc, 0] == np.nanmax(
                self.response_t[:, 1:, :, nc, 0]))
            pref_ori = cell_peak[0][0]
            pref_sf = cell_peak[1][0] + 1
            pref_phase = cell_peak[2][0]
            peak_t.cell_specimen_id.iloc[nc] = cids[nc]
            peak_t.ori_sg[nc] = pref_ori
            peak_t.sf_sg[nc] = pref_sf
            peak_t.phase_sg[nc] = pref_phase
            peak_t.response_reliability_sg[nc] = self.response_t[
                pref_ori, pref_sf, pref_phase, nc, 2] / 0.48  # TODO: check number of trials
            pref = self.response_t[pref_ori, pref_sf, pref_phase, nc, 0]
            orth = self.response_t[
                np.mod(pref_ori + 3, 6), pref_sf, pref_phase, nc, 0]
            peak_t.osi_sg[nc] = (pref - orth) / (pref + orth)
            peak_t.peak_dff_sg[nc] = pref
            groups = []

            for ori in self.orivals:
                for sf in self.sfvals[1:]:
                    #groups.append(self.mean_sweep_response_t[(self.stim_table_t.spatial_frequency == sf) & (
                    #    self.stim_table_t.orientation == ori)][str(nc)])
            #groups.append(self.mean_sweep_response_t[
            #    self.stim_table_t.spatial_frequency == 0][str(nc)])

                    # IGNORING PHASE ABOVE; TO INCLUDE PHASE USE BELOW
                    for phase in self.phasevals:
                        groups.append(self.mean_sweep_response_t[(self.stim_table_t.spatial_frequency == sf) & (
                            self.stim_table_t.orientation == ori) & (self.stim_table_t.phase == phase)][str(nc)])
            groups.append(self.mean_sweep_response_t[
                          self.stim_table_t.spatial_frequency == 0][str(nc)])

            _, p = st.f_oneway(*groups)
            peak_t.ptest_sg[nc] = p

            test_rows = (self.stim_table_t.orientation == self.orivals[pref_ori]) & \
                (self.stim_table_t.spatial_frequency == self.sfvals[pref_sf]) & \
                (self.stim_table_t.phase == self.phasevals[pref_phase])

            if len(test_rows) < 2:
                msg = "Static grating p value requires at least 2 trials at the preferred "
                "orientation/spatial frequency/phase. Cell %d (%f, %f, %f) has %d." % \
                    (int(nc), self.orivals[pref_ori], self.sfvals[pref_sf],
                     self.phasevals[pref_phase], len(test_rows))

                raise BrainObservatoryAnalysisException(msg)

            test = self.sweep_response_t[test_rows][str(nc)].mean()
            peak_t.time_to_peak_sg[nc] = (
                np.argmax(test) - self.interlength) / self.acquisition_rate
            test2 = np.where(test < (test.max() / 2))[0]
            try:
                peak_t.duration_sg[nc] = np.ediff1d(
                    test2).max() / self.acquisition_rate
            except:
                pass

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
