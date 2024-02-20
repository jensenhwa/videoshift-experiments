DATASETS = {
    # module_name, name, aliases
    'HomeActionGenomeActivities': ('dataset.homage', 'Home Action Genome Activities', ('homageactivities',)),
    'HomeActionGenomeAtomicActions': ('dataset.homage', 'Home Action Genome Atomic Actions', ('homageactions',)),
    'InteractADLActivities': ('dataset.interactadl', 'InteractADL Activities', ('interactadlactivities',)),
    'InteractADLAtomicActions': ('dataset.interactadl', 'InteractADL Atomic Actions', ('interactadlactions',)),
    'MetaverseAtWorkActivities': ('dataset.metaverse', 'Metaverse@Work Activities', ('metaverseactivities',)),
    'MetaverseAtWorkAtomicActions': ('dataset.metaverse', 'Metaverse@Work Atomic Actions', ('metaverseactions',)),
}

_dataset_cache = {}


def _load_datasets(module_name):
    """Load a dataset (and all others in the module too)."""
    mod = __import__(module_name, None, None, ['__all__'])
    for dataset_name in mod.__all__:
        cls = getattr(mod, dataset_name)
        _dataset_cache[cls.name] = cls


def get_dataset_by_name(_alias, **options):
    """
    Return an instance of a `Lexer` subclass that has `alias` in its
    aliases list. The dataset is given the `options` at its
    instantiation.
    """
    if not _alias:
        raise ValueError('no dataset for alias %r found' % _alias)

    for module_name, name, aliases in DATASETS.values():
        if _alias.lower() in aliases:
            if name not in _dataset_cache:
                _load_datasets(module_name)
            return _dataset_cache[name](**options)
    raise ValueError('no dataset for alias %r found' % _alias)
