import operator

directory = '/mdspace/pyrod/examples/4n6h/screening/pyrod/library'
number_of_actives = 296
number_of_decoys = 849
sdf_files = ['.'.join([str(x), 'sdf']) for x in range(4251)]


def sdf_parser(sdf_path):
    actives_indices = []
    actives_scores = []
    decoys_scores = []
    hit_type_pointer = False
    index_pointer = False
    score_pointer = False
    hit_type = None
    index = None
    score = None
    with open(sdf_path, 'r') as sdf_text:
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


with open('/'.join([directory, 'enrichment_factors.txt']), 'w') as result_file:
    with open('/'.join([directory, 'enrichment_factors_no_hits.txt']), 'w') as no_hits_file:
        lines = []
        header = ['file', 'EF1', 'EF5', 'EF10', 'EF100', 'number of active hits', 'active hits indices']
        no_hits = []
        actives_indices_collector = []
        for sdf_file in sdf_files:
            actives_scores, decoys_scores, actives_indices = sdf_parser('/'.join([directory, sdf_file]))
            if len(actives_scores + decoys_scores) > 0:
                lines.append([
                    sdf_file,
                    enrichment_factor(actives_scores, decoys_scores, number_of_actives, number_of_decoys, 0.01),
                    enrichment_factor(actives_scores, decoys_scores, number_of_actives, number_of_decoys, 0.05),
                    enrichment_factor(actives_scores, decoys_scores, number_of_actives, number_of_decoys, 0.1),
                    enrichment_factor(actives_scores, decoys_scores, number_of_actives, number_of_decoys, 1),
                    len(actives_indices),
                    actives_indices])
                actives_indices_collector += actives_indices
            else:
                no_hits.append(sdf_file)
        lines = sorted(lines, key=operator.itemgetter(5), reverse=True)
        lines = ['\t'.join(map(str, line)) for line in [header] + lines]
        result_file.write('\n'.join(lines))
        no_hits_file.write('\n'.join(no_hits))
        print('All pharmacophores found {} % of actives.'.format(round((len(set(actives_indices_collector)) /
                                                                 number_of_actives) * 100), 1))
