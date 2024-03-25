from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
from scipy import stats
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis, PCA, NMF

"""Decode conditions from imported Spyglass data

Uses Scikit-learn classifiers to decode the trial condition from imported MatLab class. 

TODO: Make the code more flexible so that it can decode arbitrary conditions on an arbitrary selection of conditions. 

TODO: Allow training and testing of the same decoder on different subsets of data. 

TODO: flesh out documentation of subfunctions. 

TODO: plot the output of the repeated results for the balanced splits with confidence intervals.
TODO: decide how many repetitions with the label balancing is acceptable. 
    Is there a measure which would be most resistant to an inflated sample size? 

TODO: check MatLab data import works and add sample_trials which selects a subset of trials as a method to that class. 

TODO: Generalise balance_labels to more than 2 groups. 
"""


def run_decoder(data, labels, method='Logistic', test_only_data=None, test_only_labels=None):
    """Gets cross-validated scores for the chosen classifier on both the training data and out of sample data."""
    scores = []
    test_only_scores = []

    coefficients = np.empty((5, np.shape(data)[1]))

    class_dict = {
        'Logistic': LogisticRegression(max_iter=1000),
        'SVM': svm.SVC(kernel='linear'),
        'rbfSVM': svm.SVC(kernel='rbf'),
        'Nearest': KNeighborsClassifier(3)
    }

    classifier = class_dict[method]

    skf = StratifiedKFold()
    counter = 0
    for train_index, test_index in skf.split(data, labels):
        data_train, data_test = data[train_index], data[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]

        classifier.fit(data_train, labels_train)
        score = classifier.score(data_test, labels_test)
        scores.append(score)

        if test_only_data is not None:
            individual_score = classifier.score(test_only_data, test_only_labels)
            test_only_scores.append(individual_score)

        # Todo: maybe permanently remove this bit of code
        # coefficients[counter, :] = classifier.coef_[0]
        # counter = counter + 1
        # average_coefs = np.average(coefficients, 0)
        # classifier.coef_[0] = average_coefs

    if test_only_data is None:
        return classifier, scores
    else:
        return classifier, scores, test_only_scores


def repeat_across_time(base_data, base_labels, method="Nearest", test_only_data=None, test_only_labels=None):
    """Train the decoder on the non-laser trials - one time point at a time

    Time is assumed to be the 3rd dimension - could be changed later
    """
    test_scores = []
    average_scores = []
    sem_scores = []
    out_of_sample_scores = []
    sem_oos_scores = []
    for time_column in range(np.shape(base_data)[2]):
        time_sample = base_data[:, :, time_column]

        if test_only_data is None:
            average_clf, scores = run_decoder(time_sample, base_labels, method)
        else:
            test_time_sample = test_only_data[:, :, time_column]
            average_clf, scores, individual_scores = run_decoder(time_sample, base_labels, method, test_time_sample,
                                                                 test_only_labels)

            # CAUTION: In retrospect I don't think that averaging coefficients this way is sensible at all.
            # test_score = average_clf.score(test_time_sample, test_only_labels)
            # test_scores.append(test_score)
            # When the test data is run on each decoder in turn and the collective result is pooled.
            out_of_sample_scores.append(np.average(individual_scores))
            sem_oos_scores.append(stats.sem(individual_scores))

        average_scores.append(np.average(scores))
        sem_scores.append(stats.sem(scores))

    if test_only_data is None:
        return average_scores, sem_scores
    else:
        return average_scores, sem_scores, out_of_sample_scores, sem_oos_scores


def iterate_balanced_decoder(training_data, training_labels, balancing_tags, testing_data, testing_labels, repeat_number=20,
                             show_iteration_plots=False, method="Nearest"):
    """
    Repeatedly runs the selected classifiers across the trials time using a balanced selection of trial types.
    The aim is to train and test the classifier with the same number of trials of each type to avoid introducing bias,
    whilst repeating the procedure a sufficient number of times that any error introduced by the sampling is evened out.
    """

    trained_data_scores = np.empty((repeat_number, np.shape(training_data)[2]))
    # all_laser_test_scores = np.empty((repeat_number, np.shape(training_data)[2]))
    out_of_sample_data_scores = np.empty((repeat_number, np.shape(training_data)[2]))
    for rep in range(repeat_number):

        # Balance the non-laser trials
        balance_index = balance_labels(training_labels, balancing_tags)

        balanced_trials = training_labels[balance_index]
        balanced_sample = training_data[balance_index, :, :]

        average_scores, sem_scores, test_only_scores, sem_test_only = repeat_across_time(balanced_sample,
                                                                                         balanced_trials, method,
                                                                                         test_only_data=testing_data,
                                                                                         test_only_labels=testing_labels)

        if show_iteration_plots:
            # Plot the results
            fig = plt.figure()
            plt.errorbar(x_data_time, average_scores, yerr=sem_scores)
            plt.errorbar(x_data_time, test_only_scores, yerr=sem_test_only)
            # plt.plot(x_data_time, laser_scores)
            # plt.legend(['Laser', 'Laser ind', 'Non-laser'])
            plt.legend(['Trained data', 'Out of sample'])
            plt.ylabel('Decoder performance')
            plt.xlabel('Time (s)')
            plt.title('Iteration: ' + str(rep))
            plt.show()

        trained_data_scores[rep, :] = average_scores
        # all_laser_test_scores[rep, :] = laser_scores
        out_of_sample_data_scores[rep, :] = test_only_scores

    return trained_data_scores, out_of_sample_data_scores


def balance_labels(labels, balance_tags):
    """
    Function to fetch the same number of trials of each condition from the list provided.
    """
    balance_index = np.tile(False, np.shape(labels))

    # Grab the index of balanced tags for each subcontrast. 
    unique_labels, counts = np.unique(labels, return_counts=True)
    unique_contrasts  = np.unique(balance_tags)
    for con in unique_contrasts:
        miss_con = np.logical_and(labels == unique_labels[0], balance_tags==con)
        hit_con = np.logical_and(labels == unique_labels[1], balance_tags==con)

        miss_num = np.sum(miss_con)
        hit_num = np.sum(hit_con)

        if miss_num == 0 or hit_num == 0:
            continue
        elif miss_num > hit_num:
            balance_index[hit_con] = True
            miss_loc = np.where(miss_con)
            balance_index[np.random.permutation(miss_loc)[0][0:hit_num]] = True
        elif miss_num < hit_num:
            balance_index[miss_con] = True
            hit_loc = np.where(hit_con)
            balance_index[np.random.permutation(hit_loc)[0][0:miss_num]] = True


    
    # largest_group_size = np.max(counts)
    # smallest_group_size = np.min(counts)
    # smallest_group = np.where(counts == smallest_group_size)[0][0]
    # largest_group = np.where(counts == largest_group_size)[0][0]

    # trials_to_exclude = largest_group_size - smallest_group_size
    
    # large_group_index = np.where(labels == unique_labels[largest_group])[0]
    # if trials_to_exclude == 0:
    #     return balance_index
    # else:
    #     balance_index[np.random.permutation(large_group_index)[0:trials_to_exclude]] = False

    return balance_index


def sample_trials(model_data, filter_labels, chosen_trials, output_labels=None):
    for ii in range(np.shape(chosen_trials)[0]):
        if ii == 0:
            trial_index = filter_labels == chosen_trials[ii]
        else:
            trial_index = np.logical_or(trial_index, filter_labels == chosen_trials[ii])

    if output_labels is None:
        filtered_output_labels = filter_labels[trial_index]
    else:
        filtered_output_labels = output_labels[trial_index]

    filtered_model_data = model_data[trial_index, :, :]
    unused_labels, trial_counts = np.unique(filtered_output_labels, return_counts=True)

    return filtered_model_data, filtered_output_labels, trial_counts

def reshape_data(neural_data):
    num_of_trials = np.shape(neural_data)[0]
    num_of_neurons = np.shape(neural_data)[1]
    num_of_timepoints = np.shape(neural_data)[2]

    print(f'Array is {num_of_trials} trials, {num_of_neurons} neurons, {num_of_timepoints} time-points. Reshaping...')

    reshaped_data = np.empty((num_of_trials * num_of_timepoints, num_of_neurons))
    for neuron in range(num_of_neurons):
        reshaped_data[:, neuron] = np.ndarray.reshape(neural_data[:, neuron, :], (num_of_timepoints * num_of_trials))

    return reshaped_data

def normalise_data(data_to_norm):
    num_of_neurons = np.shape(data_to_norm)[1]

    scaled_data = np.empty(np.shape(data_to_norm))
    for neuron in range(num_of_neurons):
        scaler = StandardScaler()
        fit_neuron = scaler.fit_transform(data_to_norm[:, neuron].reshape(-1, 1))
        scaled_data[:, neuron] = fit_neuron.reshape(np.size(fit_neuron))

    return scaled_data


def basic_plot(time_data, non_laser, laser, title_tag, plot_save_path, pVal='', pval_list=[]):
    # Plot the results
    fig = plt.figure()

    # plt.errorbar(x_data_time, np.average(vis1_non_laser_scores, 0), yerr=stats.sem(vis1_non_laser_scores, 0))
    # #plt.errorbar(x_data_time, np.average(vis1_laser_ind, 0), yerr=stats.sem(vis1_laser_ind, 0))
    # plt.errorbar(x_data_time, np.average(vis1_laser_scores, 0), yerr=stats.sem(vis1_laser_scores, 0))

    plt.errorbar(time_data, np.average(non_laser, 0), yerr=np.std(non_laser, 0))
    plt.errorbar(time_data, np.average(laser, 0), yerr=np.std(laser, 0))
    if len(pval_list) != 0:
        for tt in range(len(time_data)):
            plt.text(time_data[tt], 0.90, pval_list[tt])

    plt.legend(['Non-laser', 'Laser'], loc='upper left')
    plt.ylabel('Decoder performance')
    plt.xlabel('Time (s)')
    plt.title(title_tag + ' ' + str(pVal) + ' - Average performance')
    save_p = plot_save_path + r'\average_decoding_performance_' + title_tag
    fig.savefig(save_p)

def convert_to_sig_tags(pvals):
    sig_tags = []
    for ss in range(len(pvals)):
        if pvals[ss] < 0.001:
            sig_tags.append('***')
        elif pvals[ss] < 0.01:
            sig_tags.append('**')
        elif pvals[ss] < 0.05:
            sig_tags.append('*')
        else:
            sig_tags.append('n.s')

    return sig_tags

def run_model(neural_data, trial_number, time_number, number_components, model_choice='PCA'):
    
    if model_choice == 'PCA':
        print('Running PCA.')
        chosen_model = PCA(n_components=number_components).fit(neural_data)
    elif model_choice == 'NMF':
        print('Running NMF.')
        smallest_value = neural_data.min()
        neural_data = neural_data + abs(smallest_value)
        chosen_model = NMF(n_components=number_components, max_iter=10000).fit(neural_data)

    print('Transforming data.')
    transformed_data = chosen_model.fit_transform(neural_data)

    reshaped_transform = np.empty((trial_number, number_components, time_number))
    for dim in range(number_components):
        reshaped_transform[:, dim, :] = np.ndarray.reshape(transformed_data[:, dim], (trial_number, time_number))

    return reshaped_transform, model_choice, chosen_model
