""" PyRod - dynamic molecular interaction fields (dMIFs), based on tracing water molecules in MD simulations.

Released under the GNU Public Licence v2.

This module contains dictionaries, tuples, cutoffs and more used by by different pyrod modules.
"""

# python standard library
import shutil


########################################################################################################################
# Section for defining major interaction cutoffs. Can be adjusted by the user (hb - hydrogen bonds, hi - hydrophobic   #
# interaction, ii - ionizable interaction, ai - aromatic interaction, metal - metal interactions).                     #
########################################################################################################################
""" Maximal distances for interaction types. """
sel_cutoff_dict = {'hb': 3.9, 'hi': 5, 'ii': 6, 'ai': 6, 'metal': 3}
""" Maximal distance between heavy atoms of hydrogen bonds. """
hb_dist_dict = {'O': 3.2, 'N': 3.3, 'S': 3.9}  # (Bissantz 2010, Mills 1996, Zhou 2009)
""" Maximal angle between acceptor, hydrogen and donor atom in hydrogen bonds. """
hb_angl_dict = {'O': 130, 'N': 130, 'S': 130}  # (Bissantz 2010, Mills 1996, Zhou 2009)
""" Maximal angle for interaction of aromatic center with positive ionizable. """
CATION_PI_ANGLE_CUTOFF = 30  # (Marshall 2009)
""" Minimal allowed distance between two atoms. """
CLASH_CUTOFF = 1.5  # corresponds to the distance between hydrogen and acceptor in a short hydrogen bond
########################################################################################################################
# Section for handling of topology. Can be adjusted by the user to include e.g. co-factors and other water models in   #
# analysis.                                                                                                            #
########################################################################################################################
""" Dictionary for renaming residues to their standard residue name. Can be extended to standardize other residue names.
Tuples as keys for dictionaries need at least two entries, thus a dummy entry needs to be included if only one element 
is available (see standard_atomnames_dict). """
standard_resnames_dict = {('ARN', 'dummy'): 'ARG',
                          ('ASH', 'dummy'): 'ASP',
                          ('CYM', 'CYX'): 'CYS',
                          ('GLH', 'dummy'): 'GLU',
                          ('HSP', 'HSH', 'HIP', 'HIH', 'HID', 'HIE', 'HSD', 'HSE'): 'HIS',
                          ('H20', 'WAT', 'SOL', 'TIP3', 'TIP4', 'TIP5', 'T3P', 'T4P', 'T5P', 'SPC'): 'HOH',
                          ('LYN', 'dummy'): 'LYS'}
""" Tuple of protein residue names. Can be extended to include other residues. """
protein_resnames = ('ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',
                    'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL')
""" Tuple of valid residue names considered for interactions. Can be extended to include other residues, e.g. co-factors
or ligands. """
valid_resnames = ('ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',
                  'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL')
""" Dictionary to standardize atom names of valid residues. Can be extended if new valid residue were defined. Tuples as 
keys for dictionaries need at least two entries, thus a dummy entry needs to be included if only one element is 
available. """
standard_atomnames_dict = {'ALA': {('HN', 'H1', '1H', 'HN1', 'HT1'): 'H',
                                   ('2H', 'HN2', 'HT2'): 'H2',
                                   ('3H', 'HN3', 'HT3'): 'H3',
                                   ('O1', 'OT1', 'OCT1', 'OC1'): 'O',
                                   ('O2', 'OT', 'OT2', 'OCT2', 'OC2'): 'OXT',
                                   ('HO', 'dummy'): 'HXT'},
                           'ARG': {('HN', 'H1', '1H', 'HN1', 'HT1'): 'H',
                                   ('2H', 'HN2', 'HT2'): 'H2',
                                   ('3H', 'HN3', 'HT3'): 'H3',
                                   ('O1', 'OT1', 'OCT1', 'OC1'): 'O',
                                   ('O2', 'OT', 'OT2', 'OCT2', 'OC2'): 'OXT',
                                   ('HO', 'dummy'): 'HXT',
                                   ('1HH1', 'HN11'): 'HH11',
                                   ('2HH1', 'HN12'): 'HH12',
                                   ('1HH2', 'HN21'): 'HH21',
                                   ('2HH2', 'HN22'): 'HH22'},
                           'ASN': {('OD', 'dummy'): 'OD1',
                                   ('ND', 'dummy'): 'ND2',
                                   ('1HD2', 'HND1'): 'HD21',
                                   ('2HD2', 'HND2'): 'HD22'},
                           'ASP': {('HN', 'H1', '1H', 'HN1', 'HT1'): 'H',
                                   ('2H', 'HN2', 'HT2'): 'H2',
                                   ('3H', 'HN3', 'HT3'): 'H3',
                                   ('O1', 'OT1', 'OCT1', 'OC1'): 'O',
                                   ('O2', 'OT', 'OT2', 'OCT2', 'OC2'): 'OXT',
                                   ('HO', 'dummy'): 'HXT'},
                           'CYS': {('HN', 'H1', '1H', 'HN1', 'HT1'): 'H',
                                   ('2H', 'HN2', 'HT2'): 'H2',
                                   ('3H', 'HN3', 'HT3'): 'H3',
                                   ('O1', 'OT1', 'OCT1', 'OC1'): 'O',
                                   ('O2', 'OT', 'OT2', 'OCT2', 'OC2'): 'OXT',
                                   ('HO', 'dummy'): 'HXT',
                                   ('SG1', 'dummy'): 'SG',
                                   ('HG1', 'dummy'): 'HG'},
                           'GLN': {('HN', 'H1', '1H', 'HN1', 'HT1'): 'H',
                                   ('2H', 'HN2', 'HT2'): 'H2',
                                   ('3H', 'HN3', 'HT3'): 'H3',
                                   ('O1', 'OT1', 'OCT1', 'OC1'): 'O',
                                   ('O2', 'OT', 'OT2', 'OCT2', 'OC2'): 'OXT',
                                   ('HO', 'dummy'): 'HXT',
                                   ('OE', 'dummy'): 'OE1',
                                   ('NE', 'dummy'): 'NE2',
                                   ('1HE2', 'HNE1'): 'HE21',
                                   ('2HE2', 'HNE2'): 'HE22'},
                           'GLU': {('HN', 'H1', '1H', 'HN1', 'HT1'): 'H',
                                   ('2H', 'HN2', 'HT2'): 'H2',
                                   ('3H', 'HN3', 'HT3'): 'H3',
                                   ('O1', 'OT1', 'OCT1', 'OC1'): 'O',
                                   ('O2', 'OT', 'OT2', 'OCT2', 'OC2'): 'OXT',
                                   ('HO', 'dummy'): 'HXT'},
                           'GLY': {('HN', 'H1', '1H', 'HN1', 'HT1'): 'H',
                                   ('2H', 'HN2', 'HT2'): 'H2',
                                   ('3H', 'HN3', 'HT3'): 'H3',
                                   ('O1', 'OT1', 'OCT1', 'OC1'): 'O',
                                   ('O2', 'OT', 'OT2', 'OCT2', 'OC2'): 'OXT',
                                   ('HO', 'dummy'): 'HXT'},
                           'HIS': {('HN', 'H1', '1H', 'HN1', 'HT1'): 'H',
                                   ('2H', 'HN2', 'HT2'): 'H2',
                                   ('3H', 'HN3', 'HT3'): 'H3',
                                   ('O1', 'OT1', 'OCT1', 'OC1'): 'O',
                                   ('O2', 'OT', 'OT2', 'OCT2', 'OC2'): 'OXT',
                                   ('HO', 'dummy'): 'HXT',
                                   ('HND', 'HND1'): 'HD1',
                                   ('HNE', 'HNE2'): 'HE2'},
                           'HOH': {('HN', 'H1', '1H', 'HN1', 'HT1'): 'H',
                                   ('2H', 'HN2', 'HT2'): 'H2',
                                   ('3H', 'HN3', 'HT3'): 'H3',
                                   ('O1', 'OT1', 'OCT1', 'OC1'): 'O',
                                   ('O2', 'OT', 'OT2', 'OCT2', 'OC2'): 'OXT',
                                   ('HO', 'dummy'): 'HXT',
                                   ('OW', 'OH2', 'O1'): 'O',
                                   ('HW1', 'dummy'): 'H1',
                                   ('HW2', 'dummy'): 'H2'},
                           'ILE': {('HN', 'H1', '1H', 'HN1', 'HT1'): 'H',
                                   ('2H', 'HN2', 'HT2'): 'H2',
                                   ('3H', 'HN3', 'HT3'): 'H3',
                                   ('O1', 'OT1', 'OCT1', 'OC1'): 'O',
                                   ('O2', 'OT', 'OT2', 'OCT2', 'OC2'): 'OXT',
                                   ('HO', 'dummy'): 'HXT',
                                   ('CD', 'dummy'): 'CD1'},
                           'LEU': {('HN', 'H1', '1H', 'HN1', 'HT1'): 'H',
                                   ('2H', 'HN2', 'HT2'): 'H2',
                                   ('3H', 'HN3', 'HT3'): 'H3',
                                   ('O1', 'OT1', 'OCT1', 'OC1'): 'O',
                                   ('O2', 'OT', 'OT2', 'OCT2', 'OC2'): 'OXT',
                                   ('HO', 'dummy'): 'HXT'},
                           'LYS': {('HN', 'H1', '1H', 'HN1', 'HT1'): 'H',
                                   ('2H', 'HN2', 'HT2'): 'H2',
                                   ('3H', 'HN3', 'HT3'): 'H3',
                                   ('O1', 'OT1', 'OCT1', 'OC1'): 'O',
                                   ('O2', 'OT', 'OT2', 'OCT2', 'OC2'): 'OXT',
                                   ('HO', 'dummy'): 'HXT',
                                   ('1HZ', 'HNZ1'): 'HZ1',
                                   ('2HZ', 'HNZ2'): 'HZ2',
                                   ('3HZ', 'HNZ3'): 'HZ3'},
                           'MET': {('HN', 'H1', '1H', 'HN1', 'HT1'): 'H',
                                   ('2H', 'HN2', 'HT2'): 'H2',
                                   ('3H', 'HN3', 'HT3'): 'H3',
                                   ('O1', 'OT1', 'OCT1', 'OC1'): 'O',
                                   ('O2', 'OT', 'OT2', 'OCT2', 'OC2'): 'OXT',
                                   ('HO', 'dummy'): 'HXT'},
                           'PHE': {('HN', 'H1', '1H', 'HN1', 'HT1'): 'H',
                                   ('2H', 'HN2', 'HT2'): 'H2',
                                   ('3H', 'HN3', 'HT3'): 'H3',
                                   ('O1', 'OT1', 'OCT1', 'OC1'): 'O',
                                   ('O2', 'OT', 'OT2', 'OCT2', 'OC2'): 'OXT',
                                   ('HO', 'dummy'): 'HXT'},
                           'PRO': {('HN', 'H1', '1H', 'HN1', 'HT1'): 'H',
                                   ('2H', 'HN2', 'HT2'): 'H2',
                                   ('3H', 'HN3', 'HT3'): 'H3',
                                   ('O1', 'OT1', 'OCT1', 'OC1'): 'O',
                                   ('O2', 'OT', 'OT2', 'OCT2', 'OC2'): 'OXT',
                                   ('HO', 'dummy'): 'HXT'},
                           'SER': {('HN', 'H1', '1H', 'HN1', 'HT1'): 'H',
                                   ('2H', 'HN2', 'HT2'): 'H2',
                                   ('3H', 'HN3', 'HT3'): 'H3',
                                   ('O1', 'OT1', 'OCT1', 'OC1'): 'O',
                                   ('O2', 'OT', 'OT2', 'OCT2', 'OC2'): 'OXT',
                                   ('HO', 'dummy'): 'HXT',
                                   ('OG1', 'dummy'): 'OG',
                                   ('HG1', 'HOG'): 'HG'},
                           'THR': {('HN', 'H1', '1H', 'HN1', 'HT1'): 'H',
                                   ('2H', 'HN2', 'HT2'): 'H2',
                                   ('3H', 'HN3', 'HT3'): 'H3',
                                   ('O1', 'OT1', 'OCT1', 'OC1'): 'O',
                                   ('O2', 'OT', 'OT2', 'OCT2', 'OC2'): 'OXT',
                                   ('HO', 'dummy'): 'HXT',
                                   ('OG', 'dummy'): 'OG1',
                                   ('CG', 'dummy'): 'CG2',
                                   ('HOG', 'HOG1', '1HG'): 'HG1'},
                           'TRP': {('HN', 'H1', '1H', 'HN1', 'HT1'): 'H',
                                   ('2H', 'HN2', 'HT2'): 'H2',
                                   ('3H', 'HN3', 'HT3'): 'H3',
                                   ('O1', 'OT1', 'OCT1', 'OC1'): 'O',
                                   ('O2', 'OT', 'OT2', 'OCT2', 'OC2'): 'OXT',
                                   ('HO', 'dummy'): 'HXT',
                                   ('HNE', 'dummy'): 'HE1'},
                           'TYR': {('HN', 'H1', '1H', 'HN1', 'HT1'): 'H',
                                   ('2H', 'HN2', 'HT2'): 'H2',
                                   ('3H', 'HN3', 'HT3'): 'H3',
                                   ('O1', 'OT1', 'OCT1', 'OC1'): 'O',
                                   ('O2', 'OT', 'OT2', 'OCT2', 'OC2'): 'OXT',
                                   ('HO', 'dummy'): 'HXT',
                                   ('HOH', 'dummy'): 'HH'},
                           'VAL': {('HN', 'H1', '1H', 'HN1', 'HT1'): 'H',
                                   ('2H', 'HN2', 'HT2'): 'H2',
                                   ('3H', 'HN3', 'HT3'): 'H3',
                                   ('O1', 'OT1', 'OCT1', 'OC1'): 'O',
                                   ('O2', 'OT', 'OT2', 'OCT2', 'OC2'): 'OXT',
                                   ('HO', 'dummy'): 'HXT'}}
""" Dictionary to define hydrogen bond donors based on standardized residue and atom names. Can be extended if new valid 
residues were defined. Each residue contains a dictionary of donor heavy atom names and their hydrogen atom names. 
In later topology reading residues are checked if donor atoms or hydrogens actually exist (e.g. most residues in a 
protein sequence do not have the terminal OXT atom and ASP OD2 is usually not protonated. """
hd_sel_dict = {'ALA': {'N': ['H', 'H2', 'H3'],
                       'OXT': ['HXT']},
               'ARG': {'N': ['H', 'H2', 'H3'],
                       'NE': ['HE'],
                       'NH1': ['HH11', 'HH12'],
                       'NH2': ['HH21', 'HH22'],
                       'OXT': ['HXT']},
               'ASN': {'N': ['H', 'H2', 'H3'],
                       'ND2': ['HD21', 'HD22'],
                       'OXT': ['HXT']},
               'ASP': {'N': ['H', 'H2', 'H3'],
                       'OD2': ['HD2'],
                       'OXT': ['HXT']},
               'CYS': {'N': ['H', 'H2', 'H3'],
                       'SG': ['HG'],
                       'OXT': ['HXT']},
               'GLN': {'N': ['H', 'H2', 'H3'],
                       'NE2': ['HE21', 'HE22'],
                       'OXT': ['HXT']},
               'GLU': {'N': ['H', 'H2', 'H3'],
                       'OE2': ['HE2'],
                       'OXT': ['HXT']},
               'GLY': {'N': ['H', 'H2', 'H3'],
                       'OXT': ['HXT']},
               'HIS': {'N': ['H', 'H2', 'H3'],
                       'NE2': ['HE2'],
                       'ND1': ['HD1'],
                       'OXT': ['HXT']},
               'ILE': {'N': ['H', 'H2', 'H3'],
                       'OXT': ['HXT']},
               'LEU': {'N': ['H', 'H2', 'H3'],
                       'OXT': ['HXT']},
               'LYS': {'N': ['H', 'H2', 'H3'],
                       'NZ': ['HZ1', 'HZ2', 'HZ3'],
                       'OXT': ['HXT']},
               'MET': {'N': ['H', 'H2', 'H3'],
                       'OXT': ['HXT']},
               'PHE': {'N': ['H', 'H2', 'H3'],
                       'OXT': ['HXT']},
               'PRO': {'N': ['H', 'H2'],
                       'OXT': ['HXT']},
               'SER': {'N': ['H', 'H2', 'H3'],
                       'OG': ['HG'],
                       'OXT': ['HXT']},
               'THR': {'N': ['H', 'H2', 'H3'],
                       'OG1': ['HG1'],
                       'OXT': ['HXT']},
               'TRP': {'N': ['H', 'H2', 'H3'],
                       'NE1': ['HE1'],
                       'OXT': ['HXT']},
               'TYR': {'N': ['H', 'H2', 'H3'],
                       'OH': ['HH'],
                       'OXT': ['HXT']},
               'VAL': {'N': ['H', 'H2', 'H3'],
                       'OXT': ['HXT']}}
""" Dictionary to define hydrogen bond acceptors based on standardized residue and atom names. Can be extended if new 
valid residues were defined. Each residue is linked to a list of acceptor heavy atom names. In later topology reading 
residues are checked if acceptor atoms actually exist (e.g. most residues in a protein sequence do not have the terminal
OXT atom. """
ha_sel_dict = {'ALA': ['O', 'OXT'],
               'ARG': ['O', 'OXT'],
               'ASN': ['O', 'OD1', 'OXT'],
               'ASP': ['O', 'OD1', 'OD2', 'OXT'],
               'CYS': ['O', 'SG', 'OXT'],
               'GLN': ['O', 'OE1', 'OXT'],
               'GLU': ['O', 'OE1', 'OE2', 'OXT'],
               'GLY': ['O', 'OXT'],
               'HIS': ['O', 'ND1', 'NE2', 'OXT'],
               'ILE': ['O', 'OXT'],
               'LEU': ['O', 'OXT'],
               'LYS': ['O', 'OXT'],
               'MET': ['O', 'SD', 'OXT'],
               'PHE': ['O', 'OXT'],
               'PRO': ['O', 'OXT'],
               'SER': ['O', 'OG', 'OXT'],
               'THR': ['O', 'OG1', 'OXT'],
               'TRP': ['O', 'OXT'],
               'TYR': ['O', 'OH', 'OXT'],
               'VAL': ['O', 'OXT']}
""" Dictionary to define hydrophobic atoms based on standardized residue and atom names. Can be extended if new 
valid residues were defined. Each residue is linked to a list of hydrophobic heavy atom names. Default rule for defining
this dictionary was to include carbon and sulfur atoms that are not bonded to nitrogen, oxygen. Cysteine atoms ar only 
included if protonated (checked internally). """
hi_sel_dict = {'ALA': ['CB'],
               'ARG': ['CB', 'CG'],
               'ASN': ['CB'],
               'ASP': ['CB'],
               'CYS': ['CB', 'SG'],
               'GLN': ['CB', 'CG'],
               'GLU': ['CB', 'CG'],
               'HIS': ['CB'],
               'ILE': ['CB', 'CG1', 'CG2', 'CD1'],
               'LEU': ['CB', 'CG', 'CD1', 'CD2'],
               'LYS': ['CB', 'CG', 'CD'],
               'MET': ['CB', 'CG', 'SD', 'CE'],
               'PHE': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
               'PRO': ['CB', 'CG'],
               'THR': ['CG2'],
               'TRP': ['CB', 'CG', 'CD2', 'CE3', 'CZ3', 'CH2', 'CZ2'],
               'TYR': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2'],
               'VAL': ['CB', 'CG1', 'CG2']}
""" Dictionary to define negative ionizable atom groups based on standardized residue and atom names. Can be extended if
new valid residues were defined. Each residue is linked to a list of atom groups. Each atom group contains a list of the 
heavy atoms and a list of hydrogen atoms that should not be present for a charge. An atom group can consist of maximal 2
heavy atoms (e.g. ASP carboxyl group is defined by heavy atoms OD1 and OD2 and only charged if HD2 is not present). In 
case of two heavy atoms in an atom group the center between both atoms is taken as charge center."""
ni_sel_dict = {'ALA': [[['O', 'OXT'], ['HXT']]],
               'ARG': [[['O', 'OXT'], ['HXT']]],
               'ASN': [[['O', 'OXT'], ['HXT']]],
               'ASP': [[['O', 'OXT'], ['HXT']], [['OD1', 'OD2'], ['HD2']]],
               'CYS': [[['O', 'OXT'], ['HXT']], [['SG'], ['HG']]],
               'GLN': [[['O', 'OXT'], ['HXT']]],
               'GLU': [[['O', 'OXT'], ['HXT']], [['OE1', 'OE2'], ['HE2']]],
               'GLY': [[['O', 'OXT'], ['HXT']]],
               'HIS': [[['O', 'OXT'], ['HXT']]],
               'ILE': [[['O', 'OXT'], ['HXT']]],
               'LEU': [[['O', 'OXT'], ['HXT']]],
               'LYS': [[['O', 'OXT'], ['HXT']]],
               'MET': [[['O', 'OXT'], ['HXT']]],
               'PHE': [[['O', 'OXT'], ['HXT']]],
               'PRO': [[['O', 'OXT'], ['HXT']]],
               'SER': [[['O', 'OXT'], ['HXT']], [['OG'], ['HG']]],
               'THR': [[['O', 'OXT'], ['HXT']], [['OG1'], ['HG1']]],
               'TRP': [[['O', 'OXT'], ['HXT']]],
               'TYR': [[['O', 'OXT'], ['HXT']], [['OH'], ['HH']]],
               'VAL': [[['O', 'OXT'], ['HXT']]]}
""" Dictionary to define positive ionizable atom groups based on standardized residue and atom names. Can be extended if
new valid residues were defined. Each residue is linked to a list of atom groups. Each atom group contains a list of the 
heavy atoms and a list of hydrogen atoms that have to be present for a charge. An atom group can consist of maximal 2
heavy atoms (e.g. HIS imidazole group is defined by heavy atoms ND1 and NE2 and only charged if HD1 and HE2 are 
present). In case of two heavy atoms in an atom group the center between both atoms is taken as charge center. """
pi_sel_dict = {'ALA': [[['N'], ['H', 'H2', 'H3']]],
               'ARG': [[['N'], ['H', 'H2', 'H3']], [['CZ'], ['HE', 'HH11', 'HH12', 'HH21', 'HH22']]],
               'ASN': [[['N'], ['H', 'H2', 'H3']]],
               'ASP': [[['N'], ['H', 'H2', 'H3']]],
               'CYS': [[['N'], ['H', 'H2', 'H3']]],
               'GLN': [[['N'], ['H', 'H2', 'H3']]],
               'GLU': [[['N'], ['H', 'H2', 'H3']]],
               'GLY': [[['N'], ['H', 'H2', 'H3']]],
               'HIS': [[['N'], ['H', 'H2', 'H3']], [['ND1', 'NE2'], ['HD1', 'HE2']]],
               'ILE': [[['N'], ['H', 'H2', 'H3']]],
               'LEU': [[['N'], ['H', 'H2', 'H3']]],
               'LYS': [[['N'], ['H', 'H2', 'H3']], [['NZ'], ['HZ1', 'HZ2', 'HZ3']]],
               'MET': [[['N'], ['H', 'H2', 'H3']]],
               'PHE': [[['N'], ['H', 'H2', 'H3']]],
               'PRO': [[['N'], ['H', 'H2']]],
               'SER': [[['N'], ['H', 'H2', 'H3']]],
               'THR': [[['N'], ['H', 'H2', 'H3']]],
               'TRP': [[['N'], ['H', 'H2', 'H3']]],
               'TYR': [[['N'], ['H', 'H2', 'H3']]],
               'VAL': [[['N'], ['H', 'H2', 'H3']]]}
""" Dictionary to define aromatic atom groups based on standardized residue and atom names. Can be extended if new valid
residues were defined. Each residue is linked to a list of atom groups. Each atom group contains a list of three heavy 
atoms used to define the center of the aromatic ring. HIS is checked internally for protonation. """
ai_sel_dict = {'HIS': [['CG', 'CD2', 'CE1']],
               'PHE': [['CG', 'CE1', 'CE2']],
               'TRP': [['CE2', 'CE3', 'CH2'], ['CG', 'CD2', 'NE1']],
               'TYR': [['CG', 'CE1', 'CE2']]}
########################################################################################################################
# Section to define distance based scoring dictionaries involving aromatic interactions. Each key represents a         #
# distance, each value the corresponding score.                                                                        #
########################################################################################################################
pi_stacking_distance_score_dict = {3.3: 0.84, 3.4: 0.95, 3.5: 1.0, 3.6: 0.99, 3.7: 0.96, 3.8: 0.92, 3.9: 0.87,
                                   4.0: 0.83, 4.1: 0.79, 4.2: 0.76, 4.3: 0.73, 4.4: 0.7, 4.5: 0.67, 4.6: 0.65,
                                   4.7: 0.63}  # Tsuzuki 2002
t_stacking_distance_score_dict = {4.6: 0.4, 4.7: 0.6, 4.8: 0.87, 4.9: 0.95, 5.0: 1.0, 5.1: 0.99, 5.2: 0.97,
                                  5.3: 0.93, 5.4: 0.86, 5.5: 0.80, 5.6: 0.74, 5.7: 0.69, 5.8: 0.64, 5.9: 0.59,
                                  6.0: 0.54}  # Tsuzuki 2002
cation_pi_distance_score_dict = {3.1: 0.25, 3.2: 0.43, 3.3: 0.59, 3.4: 0.72, 3.5: 0.83, 3.6: 0.92, 3.7: 0.98, 3.8: 1,
                                 3.9: 0.99, 4.0: 0.97, 4.1: 0.94, 4.2: 0.9, 4.3: 0.85, 4.4: 0.8, 4.5: 0.76, 4.6: 0.72,
                                 4.7: 0.69, 4.8: 0.66, 4.9: 0.63, 5.0: 0.6, 5.1: 0.57, 5.2: 0.54, 5.3: 0.51, 5.4: 0.48,
                                 5.5: 0.45, 5.6: 0.42, 5.7: 0.39, 5.8: 0.36, 5.9: 0.33, 6.0: 0.3}  # Gallivan 2000
########################################################################################################################
# Section for internal use only, should not be changed.                                                                #
########################################################################################################################
grid_score_dict = {'x': 0, 'y': 1, 'z': 2, 'shape': 3, 'ha': 4, 'hd': 5, 'ha2': 6, 'hd2': 7, 'hda': 8, 'ni': 9,
                   'pi': 10, 'hi': 11, 'hi_norm': 12, 'ai': 13, 'tw': 14, 'h2o': 15}
grid_list_dict = {'ha': 0, 'hd': 1, 'ha2': 2, 'hd2': 3, 'hda': 4, 'ai': 5}
feature_types = ('ai', 'ha2', 'hd2', 'hda', 'ha', 'hd', 'hi', 'pi', 'ni')
__version__ = '0.7.5'  # PyRod version
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
