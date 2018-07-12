""" PyRod - dynamic molecular interaction fields (dMIFs), based on tracing water molecules in MD simulations.

This module contains dictionaries and tuples used by by different pyrod modules.
"""

# python standard library
import shutil


__version__ = '0.7.1'  # PyRod version
grid_score_dict = {'x': 0, 'y': 1, 'z': 2, 'shape': 3, 'ha': 4, 'hd': 5, 'ha2': 6, 'hd2': 7, 'hda': 8, 'ni': 9,
                   'pi': 10, 'hi': 11, 'hi_norm': 12, 'ai': 13, 'tw': 14}
grid_list_dict = {'ha_i': 0, 'hd_i': 1, 'ha2_i': 2, 'hd2_i': 3, 'hda_ia': 4, 'hda_id': 5, 'ai_i': 6}
hb_dist_dict = {'O': 3.2, 'N': 3.3, 'S': 3.9}
hb_angl_dict = {'O': 130, 'N': 130, 'S': 130}
sel_cutoff_dict = {'hb': 3.9, 'hi': 5, 'ii': 6, 'ai': 6, 'metal': 3}
hi_sel_dict = {'ALA': ['CB'], 'ARG': ['CB', 'CG'], 'ASN': ['CB'], 'ASP': ['CB'], 'GLN': ['CB', 'CG'],
               'GLU': ['CB', 'CG'], 'HIS': ['CB'], 'ILE': ['CB', 'CG1', 'CD1', 'CG2'],
               'LEU': ['CB', 'CG', 'CD1', 'CD2'], 'LYS': ['CB', 'CG', 'CD'], 'MET': ['CB', 'CG', 'SD', 'CE'],
               'PHE': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'], 'PRO': ['CB', 'CG'], 'THR': ['CG2'],
               'TRP': ['CB', 'CG', 'CD2', 'CE3', 'CZ3', 'CH2', 'CZ2'], 'TYR': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2'],
               'VAL': ['CB', 'CG1', 'CG2']}
pi_sel_dict = {'ASP': ['OD1', 'OD2'], 'GLU': ['OE1', 'OE2']}
ni_sel_dict = {'ARG': ['CZ'], 'LYS': ['NZ']}  # HIS separately
ai_sel_dict = {'PHE': [['CG', 'CE1', 'CE2']], 'TRP': [['CE2', 'CE3', 'CH2'], ['CG', 'CD2', 'NE1']],
               'TYR': [['CG', 'CE1', 'CE2']]}  # HIS separately
don_hydrogen_dict = {'ALA': {'N': ['H']},
                     'ARG': {'N': ['H'], 'NE': ['HE'], 'NH1': ['HH11', 'HH12'], 'NH2': ['HH21', 'HH22']},
                     'ASN': {'N': ['H'], 'ND2': ['HD21', 'HD22']}, 'ASP': {'N': ['H']},
                     'CYS': {'N': ['H'], 'SG': ['HG']}, 'GLN': {'N': ['H'], 'NE2': ['HE21', 'HE22']},
                     'GLU': {'N': ['H']}, 'GLY': {'N': ['H']}, 'HIS': {'N': ['H'], 'NE2': ['HE2'], 'ND1': ['HD1']},
                     'HSD': {'N': ['H'], 'NE2': ['HE2'], 'ND1': ['HD1']},
                     'HSE': {'N': ['H'], 'NE2': ['HE2'], 'ND1': ['HD1']},
                     'HSP': {'N': ['H'], 'NE2': ['HE2'], 'ND1': ['HD1']},
                     'ILE': {'N': ['H']}, 'LEU': {'N': ['H']}, 'LYS': {'N': ['H'], 'NZ': ['HZ1', 'HZ2', 'HZ3']},
                     'MET': {'N': ['H']}, 'PHE': {'N': ['H']}, 'SER': {'N': ['H'], 'OG': ['HG']},
                     'THR': {'N': ['H'], 'OG1': ['HG1']}, 'TRP': {'N': ['H'], 'NE1': ['HE1']},
                     'TYR': {'N': ['H'], 'OH': ['HH']}, 'VAL': {'N': ['H']}}
pharm_cutoff_dict = {'general': {'any': [3, 8], 'hb': [1, 6, 20], 'hi': [1, 3, 8], 'ii': [0, 1, 5], 'max_dist': 22},
                     'fragment': {'any': [3, 5], 'hb': [1, 4, 20], 'hi': [0, 2, 8], 'ii': [0, 1, 5], 'max_dist': 12}}
hb_types = ('O', 'N', 'S')
acceptors = ('O', 'OH', 'OG', 'OG1', 'OD1', 'OE1', 'OD2', 'OE2', 'ND1', 'SG', 'SD')
protein_resnames = ('ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'HSD', 'HSE', 'HSP', 'ILE', 'LEU',
                    'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL')
feature_names = ('ha', 'hd', 'ai', 'ha2', 'hd2', 'hda', 'hi', 'pi', 'ni')
""" Interaction cutoffs. """
AI_PI_ANGLE_CUTOFF = 30  # Marshall 2009
""" Scoring dictionaries. """
pi_stacking_distance_score_dict = {3.3: 0.84, 3.4: 0.95, 3.5: 1.0, 3.6: 0.99, 3.7: 0.96, 3.8: 0.92, 3.9: 0.87,
                                   4.0: 0.83, 4.1: 0.79, 4.2: 0.76, 4.3: 0.73, 4.4: 0.7, 4.5: 0.67, 4.6: 0.65, 
                                   4.7: 0.63}  # Tsuzuki 2002
t_stacking_distance_score_dict = {4.6: 0.4, 4.7: 0.6, 4.8: 0.87, 4.9: 0.95, 5.0: 1.0, 5.1: 0.99, 5.2: 0.97, 5.3: 0.93,
                                  5.4: 0.86, 5.5: 0.80, 5.6: 0.74, 5.7: 0.69, 5.8: 0.64, 5.9: 0.59, 6.0: 0.54}
                                  # Tsuzuki 2002
ai_pi_distance_score_dict = {3.1: 0.25, 3.2: 0.43, 3.3: 0.59, 3.4: 0.72, 3.5: 0.83, 3.6: 0.92, 3.7: 0.98, 3.8: 1,
                             3.9: 0.99, 4.0: 0.97, 4.1: 0.94, 4.2: 0.9, 4.3: 0.85, 4.4: 0.8, 4.5: 0.76, 4.6: 0.72,
                             4.7: 0.69, 4.8: 0.66, 4.9: 0.63, 5.0: 0.6, 5.1: 0.57, 5.2: 0.54, 5.3: 0.51, 5.4: 0.48,
                             5.5: 0.45, 5.6: 0.42, 5.7: 0.39, 5.8: 0.36, 5.9: 0.33, 6.0: 0.3}  # Gallivan 2000
terminal_width = shutil.get_terminal_size()[0]
logo = ('',
        '{:^{width}}'.format('                 |X|                   ', width=terminal_width),
        '{:^{width}}'.format('                  )__)                 ', width=terminal_width),
        '{:^{width}}'.format('             )_)  )___) ))             ', width=terminal_width),
        '{:^{width}}'.format('            )___) )____))_)            ', width=terminal_width),
        '{:^{width}}'.format('       _    )____)_____))__)           ', width=terminal_width),
        '{:^{width}}'.format('        \---__|____/|___|___-----      ', width=terminal_width),
        '{:^{width}}'.format('~~~~~~~~~\   oo  oo  oo  oo  /~~~~~~~~~', width=terminal_width),
        '{:^{width}}'.format('  ~~~~~~~~~~~~~~~~~~     ~~~~~~    ~~  ', width=terminal_width),
        '{:^{width}}'.format('     ~~~~    PyRod v{}     ~~~~~     '.format(__version__), width=terminal_width),
        '{:^{width}}'.format('           ~~~~     ~~    ~~~~         ', width=terminal_width),
        '{:^{width}}'.format(' Tracing water molecules in molecular  ', width=terminal_width),
        '{:^{width}}'.format('        dynamics simulations.          ', width=terminal_width),
        '')
help_description = 'PyRod v{} - Tracing water molecules in molecular dynamics simulations.'.format(__version__)
