""" PyRod - dynamic molecular interaction fields (dMIFs), based on tracing water molecules in MD simulations.

This module contains helper functions used for user feedback and writing to the log file.
"""


# python standard libraries
import sys
import time


def time_to_text(seconds):
    if seconds > 60:
        if seconds > 3600:
            if seconds > 86400:
                if seconds > 1209600:
                    if seconds > 31449600:
                        time_as_text = 'years'
                    else:
                        time_as_text = '{} weeks'.format(round(seconds / 1209600, 1))
                else:
                    time_as_text = '{} d'.format(round(seconds / 86400, 1))
            else:
                time_as_text = '{} h'.format(round(seconds / 3600, 1))
        else:
            time_as_text = '{} min'.format(round(seconds / 60, 1))
    else:
        time_as_text = '{} s'.format(int(seconds))
    return time_as_text


def bytes_to_text(bytes_number):
    if bytes_number > 1000:
        if bytes_number > 1000000:
            if bytes_number > 1000000000:
                if bytes_number > 1000000000000:
                    bytes_as_text = '{} TB'.format(round(bytes_number / 1000000000000, 2))
                else:
                    bytes_as_text = '{} GB'.format(round(bytes_number / 1000000000, 2))
            else:
                bytes_as_text = '{} MB'.format(round(bytes_number / 1000000, 2))
        else:
            bytes_as_text = '{} KB'.format(round(bytes_number / 1000, 2))
    else:
        bytes_as_text = '{} B'.format(bytes_number)
    return bytes_as_text


def update_progress(progress, progress_info, eta, final=True):
    """ This function writes a progress bar to the terminal. """
    bar_length = 10
    block = int(bar_length * progress)
    if progress == 1.0:
        if final:
            status = '         Done\n'
        else:
            return
    else:
        status = '  ETA {:8}'.format(time_to_text(eta))
    text = '\r{}: [{}] {:>5.1f}%{}'.format(progress_info, '=' * block + ' ' * (bar_length - block), progress * 100,
                                           status)
    sys.stdout.write(text)
    sys.stdout.flush()
    return


def update_progress_dmif_parameters(counter, length_trajectory, number_processes, number_trajectories):
    """ This function returns parameters for update_progress_dmif. """
    check_progress = False
    final = False
    past_frames = (counter // number_processes) * length_trajectory * number_processes
    future_frames = (number_trajectories - (counter + 1)) * length_trajectory
    if (counter + 1) % number_processes == 0:
        check_progress = True
    if (counter + 1) % number_trajectories == 0:
        final = True
        check_progress = True
    return [check_progress, final, past_frames, future_frames]


def update_progress_dmif(counter, frame, length_trajectory, number_trajectories, number_processes, past_frames,
                         future_frames, start, final):
    """ This function calculates parameters and passes them to update_progress. """
    progress_info = 'Progress of trajectory analysis'
    actual_frame = frame + 1
    time_per_frame = (time.time() - start) / actual_frame
    left_frames = length_trajectory - actual_frame
    total_frames = length_trajectory * number_trajectories
    if (counter + 1) % number_trajectories == 0:
        eta = time_per_frame * left_frames
    else:
        if number_processes > number_trajectories - (counter + 1):
            eta = time_per_frame * (left_frames + (future_frames / (number_trajectories - (counter + 1))))
        else:
            eta = time_per_frame * (left_frames + (future_frames / number_processes))
    if final:
        progress = ((actual_frame * (number_trajectories - ((counter // number_processes) * number_processes)))
                    + past_frames) / total_frames
    else:
        progress = ((actual_frame * number_processes) + past_frames) / total_frames
    update_progress(progress, progress_info, eta, final)
    return


def update_user(text, logger):
    """ This function writes information to the terminal and to a log file. """
    print(text)
    logger.info(text)
    return
