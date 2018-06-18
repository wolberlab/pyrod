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


def main_parameters(config):
    test_grid_generation, dmif_generation, exclusion_volume_generation, feature_generation, pharmacophore_generation, \
        library_generation = False, False, False, False, False, False
    try:
        if config.get('grid parameters', 'test grid generation') == 'true':
            test_grid_generation = True
    except KeyError:
        pass
    try:
        if config.get('dmif parameters', 'dmif generation') == 'true':
            dmif_generation = True
    except KeyError:
        pass
    try:
        if config.get('exclusion volume parameters', 'exclusion volume generation') == 'true':
            exclusion_volume_generation = True
    except KeyError:
        pass
    try:
        if config.get('feature parameters', 'feature generation') == 'true':
            feature_generation = True
    except KeyError:
        pass
    try:
        if config.get('pharmacophore parameters', 'pharmacophore generation') == 'true':
            pharmacophore_generation = True
    except KeyError:
        pass
    try:
        if config.get('library parameters', 'library generation') == 'true':
            library_generation = True
    except KeyError:
        pass
    return [test_grid_generation, dmif_generation, exclusion_volume_generation, feature_generation,
            pharmacophore_generation, library_generation]


def grid_parameters(config):
    center = [float(x.strip()) for x in config.get('grid parameters', 'center').split(',')]
    edge_lengths = [float(x.strip()) for x in config.get('grid parameters', 'edge lengths').split(',')]
    space = float(config.get('grid parameters', 'grid space'))
    name = '_'.join(str(_) for _ in center + edge_lengths + [space])
    return [center, edge_lengths, space, name]


def dmif_parameters(config):
    topology = config.get('dmif parameters', 'topology')
    trajectories = [x.strip() for x in config.get('dmif parameters', 'trajectories').split(',')]
    traj_number = len(trajectories)
    first_frame = int(config.get('dmif parameters', 'first frame')) - 1
    last_frame = None
    if len(config.get('dmif parameters', 'last frame')) > 0:
        last_frame = int(config.get('dmif parameters', 'last frame'))
    if last_frame is None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            length = len(mda.Universe(trajectories[0]).trajectory) - first_frame
    else:
        length = last_frame - first_frame
    water_name = config.get('dmif parameters', 'water name')
    metal_names = [x.strip() for x in config.get('dmif parameters', 'metal names').split(',')]
    map_formats = [x.strip() for x in config.get('dmif parameters', 'map formats').split(',')]
    mp = config.get('dmif parameters', 'number of processes')
    if len(mp) > 0:
        mp = int(mp)
    else:
        mp = 1
    return [topology, trajectories, traj_number, first_frame, last_frame, length, water_name, metal_names, map_formats,
            mp]


def exclusion_volume_parameters(config):
    ev_space = float(config.get('exclusion volume parameters', 'space'))
    shape_min_cutoff = float(config.get('exclusion volume parameters', 'shape min cutoff'))
    shape_max_cutoff = float(config.get('exclusion volume parameters', 'shape max cutoff'))
    shape_radius = float(config.get('exclusion volume parameters', 'shape radius'))
    ev_radius = float(config.get('exclusion volume parameters', 'exclusion volume radius'))
    return [ev_space, shape_min_cutoff, shape_max_cutoff, shape_radius, ev_radius]


def feature_parameters(config):
    features_per_feature_type = int(config.get('feature parameters', 'features per feature type'))
    mp = config.get('feature parameters', 'number of processes')
    if len(mp) > 0:
        mp = int(mp)
    else:
        mp = 1
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
    minimal_features = int(config.get('library parameters', 'minimal features'))
    maximal_features = int(config.get('library parameters', 'maximal features'))
    maximal_hydrogen_bonds = int(config.get('library parameters', 'maximal hydrogen bonds'))
    maximal_hydrophobic_interactions = int(config.get('library parameters', 'maximal hydrophobic interactions'))
    maximal_aromatic_interactions = int(config.get('library parameters', 'maximal aromatic interactions'))
    maximal_ionizable_interactions = int(config.get('library parameters', 'maximal ionizable interactions'))
    pyrod_pharmacophore = False
    if config.get('library parameters', 'pyrod pharmacophore') == 'true':
        pyrod_pharmacophore = True
    if len(config.get('library parameters', 'library name')) > 0:
        library_name = config.get('library parameters', 'library name')
    else:
        library_name = 'library'
    library_path = '/'.join([path, library_name])
    return [pharmacophore_path, minimal_features, maximal_features, maximal_hydrogen_bonds,
            maximal_hydrophobic_interactions, maximal_aromatic_interactions,
            maximal_ionizable_interactions,  library_path, pyrod_pharmacophore]
