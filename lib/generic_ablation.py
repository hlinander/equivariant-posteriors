import itertools


def get_config_grid(create_config, values_dict):
    """Creates a list of config factories for all combinations of values.

    Returns lazy callables compatible with run_slurm_sweep.py.

    >>> get_config_grid(create_config, dict(a=[1, 2], b=['x', 'y']))
    [lambda: create_config(a=1, b='x'), lambda: create_config(a=1, b='y'), ...]
    """
    combinations = itertools.product(*values_dict.values())
    kwarg_names = list(values_dict.keys())
    configs = []
    for values in combinations:
        kwargs = {name: val for name, val in zip(kwarg_names, values)}
        config = lambda kwargs=kwargs: create_config(**kwargs)
        configs.append(config)
    return configs
