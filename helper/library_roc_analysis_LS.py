#!/usr/bin/python
import contextlib
import operator
import os
import subprocess
import sys

last_file = 10
current_file = 0
actives = '/path/to/actives.ldb'
decoys = '/path/to/decoys.ldb'
directory = '/path/to/pharmacophore/library'
output_file_name = 'enrichment_factors.txt'
minimum_of_actives = 5  # minimum actives to be found to run a roc curve analysis in percent


def sdf_parser(sdf_path, molecule_count_only=False):
    actives_indices = []
    actives_scores = []
    decoys_scores = []
    hit_type_pointer = False
    index_pointer = False
    score_pointer = False
    hit_type = None
    index = None
    score = None
    mol_counter = 0
    with open(sdf_path, 'r') as sdf_text:
        if molecule_count_only:
            for line in sdf_text.readlines():
                if '$$$$' in line:
                    mol_counter += 1
            return mol_counter
        else:
            for line in sdf_text.readlines():
                if score_pointer:
                    score = float(line.strip())
                if hit_type_pointer:
                    hit_type = line.strip()
                if index_pointer:
                    index = line.strip()
                if '> <Mol. Index>' in line:
                    index_pointer = True
                else:
                    index_pointer = False
                if '> <Active/Decoy>' in line:
                    hit_type_pointer = True
                else:
                    hit_type_pointer = False
                if '> <Pharmacophore-Fit Score>' in line:
                    score_pointer = True
                else:
                    score_pointer = False
                if '$$$$' in line:
                    if hit_type == 'active':
                        actives_scores.append(score)
                        actives_indices.append(index)
                    else:
                        decoys_scores.append(score)
            return [sorted(actives_scores), sorted(decoys_scores), sorted(actives_indices)]


def enrichment_factor(actives_scores, decoys_scores, number_of_actives, number_of_decoys, ef_top):
    tp, fp = 0, 0
    while len(actives_scores + decoys_scores) > 0:
        if fp + tp >= (number_of_actives + number_of_decoys) * ef_top:
            break
        if len(actives_scores) > 0:
            if len(decoys_scores) > 0:
                if actives_scores[-1] >= decoys_scores[-1]:
                    tp += 1
                    actives_scores = actives_scores[:-1]
                elif actives_scores[-1] <= decoys_scores[-1]:
                    fp += 1
                    decoys_scores = decoys_scores[:-1]
            else:
                tp += 1
                actives_scores = actives_scores[:-1]
        else:
            fp += 1
            decoys_scores = decoys_scores[:-1]
    return round((tp / (tp + fp)) / (number_of_actives / (number_of_actives + number_of_decoys)), 1)


def roc_analyzer(current_file, directory, number_of_actives, number_of_decoys):
    file_name = '/'.join([directory, str(current_file)])
    if sdf_parser('.'.join([file_name, 'sdf']), True) == 0:
        return None
    actives_scores, decoys_scores, actives_indices = sdf_parser('.'.join([file_name, 'sdf']))
    EFs = []
    for EF in [0.01, 0.05, 0.1, 1]:
        EFs.append(str(enrichment_factor(actives_scores, decoys_scores, number_of_actives, number_of_decoys, EF)))
    result = [str(current_file)] + EFs + [str(len(actives_scores))] + [str(actives_indices)]
    print('\rPharmacophore {0}: EF1={1} EF5={2} EF10={3} EF100={4} actives={5}'.format(*result[:-1]))
    return result


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
           yield
        finally:
           sys.stdout = old_stdout


number_of_actives = 0
header = ['pharmacophore', 'EF1', 'EF5', 'EF10', 'EF100', 'number of active hits', 'active hits indices']
with open('/'.join([directory, output_file_name]), 'w') as result_file:
    result_file.write('\t'.join(header) + '\n')
    while current_file <= last_file:
        sys.stdout.write('\rAnalyzing pharmacophore {} of {}.'.format(current_file, last_file))
        roc_analysis = False
        file_name = '/'.join([directory, str(current_file)])
        with suppress_stdout():
            subprocess.run('iscreen -q {0}.pml -d {1}:active -o {0}.sdf -l {0}.log'.format(file_name, actives).split(),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            with open('{}.log'.format(file_name, 'r')) as log_file:
                if 'iscreen finished successfully' in log_file.read():
                    log_file.seek(0)
                    number_of_actives = 0
                    for line in log_file.readlines():
                        if 'Using database' in line:
                            try:
                                number_of_actives = int(line.split()[-2])
                                break
                            except ValueError:
                                print('\rNumber of actives in database cannot be found! Please check {}.log'.format(
                                    file_name))
                                sys.exit()
                    if number_of_actives > 0:
                        if round(number_of_actives * (minimum_of_actives / 100)) <= sdf_parser(
                                '{}.sdf'.format(file_name), True):
                            roc_analysis = True
                        else:
                            current_file += 1
                    else:
                        print('\rNumber of actives in database is found to be 0! Please check {0}.log and {1}'.format(
                            file_name, actives))
                        sys.exit()
        except FileNotFoundError:
            pass
        if roc_analysis:
            with suppress_stdout():
                subprocess.run('iscreen -q {0}.pml -d {1}:active,{2}:inactive -o {0}.sdf -l {0}.log -R {0}.png'.format(
                               file_name, actives, decoys).split(), stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            try:
                with open('{}.log'.format(file_name, 'r')) as log_file:
                    if 'iscreen finished successfully' in log_file.read():
                        log_file.seek(0)
                        database_counter = 0
                        number_of_actives = 0
                        number_of_decoys = 0
                        for line in log_file.readlines():
                            if 'Using database' in line:
                                if database_counter == 0:
                                    try:
                                        number_of_actives = int(line.split()[-2])
                                        database_counter += 1
                                    except ValueError:
                                        print('\rNumber of actives in database cannot be found! Please check '
                                              '{}.log'.format(file_name))
                                        sys.exit()
                                else:
                                    try:
                                        number_of_decoys = int(line.split()[-2])
                                        break
                                    except ValueError:
                                        print('\rNumber of decoys in database cannot be found! Please check '
                                              '{}.log'.format(file_name))
                                        sys.exit()
                        if number_of_actives > 0 and number_of_decoys > 0:
                            result = roc_analyzer(current_file, directory, number_of_actives, number_of_decoys)
                            if result is not None:
                                current_file += 1
                                result_file.write('\t'.join(result) + '\n')
                        else:
                            print('\rNumber of actives or decoys in databases is found to be 0! Please check {0}.log,' +
                                  '{1} and {2}.'.format(file_name, actives, decoys))
                            sys.exit()
            except FileNotFoundError:
                pass
