from behavior_python.detection.wheelDetectionSession import WheelDetectionSession
from behavior_python.utils import parseStimpyLog
import datajoint_pipeline.tables.main_tables as mt
import datajoint_pipeline.tables.preprocessing_tables as pt
import datajoint as dj
import os
import numpy as np
import polars as pl
import pandas as pd
from behavior_python.utils import parseStimpyLog
import matplotlib.pyplot as plt
import seaborn as sns
from twop_processing_utils import cut_data_windows
import decoding_outcome
import random

# Settings for the later decoding.
n_repeats = 100
n_neuron_sample = 50


# Fetch dF/F
df_data = (pt.NeuropilCorrection & 'session_date="2024-03-22"' & 'rec_idx=1').fetch('data_df')

# Fetch the behavioural data
behaviour_paths = (mt.Behaviour & 'session_date="2024-03-22"' & 'rec_idx=1').fetch('behave_session_path')

# Fetch the times of the imaging planes
frame_times_all = (pt.BasicRecordingData & 'session_date="2024-03-22"' & 'rec_idx=1').fetch('frame_times')

# Fetch the plane paths
plane_paths = (pt.Plane & 'session_date="2024-03-22"' & 'rec_idx=1').fetch('plane_path')


for rec in range(len(behaviour_paths)):

    w = WheelDetectionSession(sessiondir=behaviour_paths[rec], load_flag=True, skip_google=True)
    behave = w.data.data

    # Discard excess frames
    trimmed_neural = df_data[rec][:, range(np.size(frame_times_all[rec]))]

    # Fetch just the trials when the mouse responded
    print("Number of trials: %d" %(behave.shape[0]))
    stim_details = behave.drop_nulls('t_stimstart_rig')
    print("Number of trials after dropping early trials: %d" %(stim_details.shape[0]))


    # Get the unique contrasts and stimuli
    unique_labels = stim_details.unique(subset=["stim_label"]).select("stim_label")
    unique_labels = unique_labels.drop_nulls()

    # Also - sort the contrasts because initially they will come out a random order that reflects the order of the trials in which they were presented. 
    unsorted_contrasts = stim_details.unique(subset=["contrast"]).select("contrast").to_numpy()
    if len(unsorted_contrasts) > 1:
        unique_contrasts = np.sort(np.squeeze(unsorted_contrasts))
    else:
        unique_contrasts = unsorted_contrasts[0]

    unique_contrasts = unique_contrasts[~np.isnan(unique_contrasts)]


    # Get a table that has just the columns that we're interested in and an index to the trials
    indexed_details = stim_details.select(["contrast", "stim_label", "t_stimstart_rig", "outcome"])
    trial_nums = np.arange(0, np.shape(indexed_details)[0])
    indexed_details = indexed_details.with_columns(trial_num=pl.lit(trial_nums))

    # Get the data around the stimulus onset times. 
    sliced_data, times = cut_data_windows(trimmed_neural, frame_times_all[rec], stim_details.select('t_stimstart_rig').to_numpy(), 5000)
    t_sec = times/1000

    # Trial x neuron x time
    # Get the cut data and labels that we'll need to decoding
    good_outcomes = indexed_details.filter(indexed_details["outcome"] != -1)
    outcome_labels = good_outcomes.select("outcome").to_numpy().squeeze()
    outcome_trials = good_outcomes.select("trial_num").to_numpy().squeeze()
    con_label_for_outcome = good_outcomes.select('contrast').to_numpy().squeeze()

    out_slice = sliced_data[outcome_trials, :,:]
    out_slice = out_slice[:,:, np.logical_and(t_sec >=-0.5, t_sec <= 2)]


    # Set up data for decoding
    class_method = 'Logistic'

    num_of_trials = np.shape(out_slice)[0]
    num_of_neurons = np.shape(out_slice)[1]
    num_of_time_points = np.shape(out_slice)[2]

    # Reshapes the array so that it is 2D rather than 3D
    columnar_data = decoding_outcome.reshape_data(out_slice)

    # Mean normalise and scale by variance
    scaled_data = decoding_outcome.normalise_data(columnar_data)
    scaled_data[np.isnan(scaled_data)] = 0

    # Loop through different subsets of 100 neurons and see get the average of the decoding performance
    repeat_results = []
    for rep in range(n_repeats):
        print('Random repeat %d' %(rep))
        random_neurons = random.sample(range(num_of_neurons), n_neuron_sample)
        random_data = scaled_data[:,random_neurons]

        PCA_data, model_name, dr_model = decoding_outcome.run_model(scaled_data[:,random_neurons], num_of_trials, num_of_time_points, 30, 'PCA')

        outcome_scores, full_sample_score = decoding_outcome.iterate_balanced_decoder(training_data=PCA_data, training_labels=outcome_labels, balancing_tags=con_label_for_outcome, testing_data=PCA_data, testing_labels=outcome_labels, repeat_number=50,
                                show_iteration_plots=False, method=class_method)
        
        repeat_results.append(np.mean(outcome_scores, axis=0))


    decoding_results_path = os.path.join(plane_paths[0], 'outcome_decoding_50_neurons_contrast_balanced.npy')
    repeat_results = np.array(repeat_results)
    print('Saving results to %s' %(decoding_results_path))
    np.save(decoding_results_path, repeat_results)

