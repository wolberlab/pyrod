""" PyRod - dynamic molecular interaction fields (dMIFs), based on tracing water molecules in MD simulations.

This is the main script to run PyRod from the command line for analyzing molecular dynamics simulations and generating
dMIFs and pharmacophores.
"""

# python standard libraries
import warnings

# external libraries
import MDAnalysis as mda

# pyrod modules
try:
    from pyrod.modules.helper_update import update_user
    from pyrod.modules.lookup import feature_names
except ImportError:
    from modules.helper_update import update_user
    from modules.lookup import feature_names


def test_grid_parameters(config):
    center = [int(x.strip()) for x in config.get('test grid parameters', 'center').split(',')]
    edge_lengths = [int(x.strip()) for x in config.get('test grid parameters', 'edge lengths').split(',')]
    name = '_'.join(str(_) for _ in center + edge_lengths)
    return [center, edge_lengths, name]


def trajectory_analysis_parameters(config):
    center = [int(x.strip()) for x in config.get('trajectory analysis parameters', 'center').split(',')]
    edge_lengths = [int(x.strip()) for x in config.get('trajectory analysis parameters', 'edge lengths').split(',')]
    topology = config.get('trajectory analysis parameters', 'topology')
    trajectories = [x.strip() for x in config.get('trajectory analysis parameters', 'trajectories').split(',')]
    traj_number = len(trajectories)
    first_frame = 0
    if len(config.get('trajectory analysis parameters', 'first frame')) > 0:
        first_frame = int(config.get('trajectory analysis parameters', 'first frame')) - 1
    last_frame = None
    if len(config.get('trajectory analysis parameters', 'last frame')) > 0:
        last_frame = int(config.get('trajectory analysis parameters', 'last frame'))
    if last_frame is None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            length = len(mda.Universe(trajectories[0]).trajectory) - first_frame
    else:
        length = last_frame - first_frame
    metal_names = [x.strip() for x in config.get('trajectory analysis parameters', 'metal names').split(',')]
    map_formats = []
    if len(config.get('trajectory analysis parameters', 'map formats')) > 0:
        map_formats = [x.strip() for x in config.get('trajectory analysis parameters', 'map formats').split(',')]
    mp = int(config.get('trajectory analysis parameters', 'number of processes'))
    get_partners = True
    if config.has_option('trajectory analysis parameters', 'dmifs only'):
        if config.get('trajectory analysis parameters', 'dmifs only') == 'true':
            get_partners = False
    return [center, edge_lengths, topology, trajectories, traj_number, first_frame, last_frame, length, metal_names,
            map_formats, mp, get_partners]


def exclusion_volume_parameters(config):
    shape_cutoff = float(config.get('exclusion volume parameters', 'shape cutoff'))
    restrictive = config.get('exclusion volume parameters', 'restrictive')
    if restrictive == 'true':
        restrictive = True
    else:
        restrictive = False
    return [shape_cutoff, restrictive]


def feature_parameters(config):
    features_per_feature_type = int(config.get('feature parameters', 'features per feature type'))
    mp = int(config.get('feature parameters', 'number of processes'))
    return [features_per_feature_type, mp]


def pharmacophore_parameters(config):
    pharmacophore_formats = [x.strip() for x in config.get('pharmacophore parameters',
                                                           'pharmacophore formats').split(',')]
    if config.get('pharmacophore parameters', 'weight') == 'true':
        weight = True
    else:
        weight = False
    if config.get('pharmacophore parameters', 'all') == 'true':
        all_features = True
    else:
        all_features = False
    if config.get('pharmacophore parameters', 'best') == 'true':
        best_features = True
        best_name = config.get('pharmacophore parameters', 'best name')
        hbs_number = int(config.get('pharmacophore parameters', 'hydrogen bonding features'))
        his_number = int(config.get('pharmacophore parameters', 'hydrophobic features'))
        iis_number = int(config.get('pharmacophore parameters', 'ionizable features'))
        ais_number = int(config.get('pharmacophore parameters', 'aromatic features'))
    else:
        best_features = False
        best_name = None
        hbs_number = None
        his_number = None
        iis_number = None
        ais_number = None
    return [pharmacophore_formats, weight, all_features, best_features, best_name, hbs_number, his_number, iis_number,
            ais_number]


def library_parameters(config, path):
    pharmacophore_path = config.get('library parameters', 'pharmacophore path')
    library_dict = {}
    for parameter in ['minimal features', 'maximal features', 'minimal hydrogen bonds', 'maximal hydrogen bonds',
                      'minimal hydrophobic interactions', 'maximal hydrophobic interactions', 
                      'minimal aromatic interactions', 'maximal aromatic interactions', 
                      'minimal ionizable interactions', 'maximal ionizable interactions']:
        library_dict[parameter] = int(config.get('library parameters', parameter))
    make_mandatory = False
    if config.get('library parameters', 'make optional features mandatory') == 'true':
        make_mandatory = True
    if len(config.get('library parameters', 'library name')) > 0:
        library_name = config.get('library parameters', 'library name')
    else:
        library_name = 'library'
    library_path = '/'.join([path, library_name])
    pyrod_pharmacophore = True
    if config.get('library parameters', 'pyrod pharmacophore') == 'false':
        pyrod_pharmacophore = False
    return [pharmacophore_path, library_dict, library_path, make_mandatory, pyrod_pharmacophore]


def dmif_excess_parameters(config):
    dmif1_path = config.get('dmif excess parameters', 'dmif 1')
    dmif2_path = config.get('dmif excess parameters', 'dmif 2')
    dmif1_name = config.get('dmif excess parameters', 'dmif 1 name')
    if len(dmif1_name) == 0:
        dmif1_name = 'dmif1'
    dmif2_name = config.get('dmif excess parameters', 'dmif 2 name')
    if len(dmif2_name) == 0:
        dmif2_name = 'dmif2'
    map_formats = []
    if len(config.get('dmif excess parameters', 'map formats')) > 0:
        map_formats = [x.strip() for x in config.get('dmif parameters', 'map formats').split(',')]
    return [dmif1_path, dmif2_path, dmif1_name, dmif2_name, map_formats]
