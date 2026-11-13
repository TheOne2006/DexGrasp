"""CPU-only scheduling policy; continuous runs have no implicit deadline."""
import itertools


def validate_schedule(config):
    continuous = config.get('continuous', False)
    hard = config.get('hard_deadline_epoch')
    stop = config.get('stop_new_init_epoch')
    if continuous:
        assert hard is None and stop is None, 'Continuous mode cannot inherit a deadline'
        assert config.get('tours') is None and config.get('max_group_visits') is None
    else:
        assert isinstance(config['tours'], int) and config['tours'] > 0
        assert hard is not None and stop is not None and hard > stop
    return continuous


def tour_numbers(config, start=0):
    validate_schedule(config)
    return itertools.count(start) if config.get('continuous') else range(start, config['tours'])


def planned_visits(config):
    if config.get('continuous'):
        return None
    total = config['tours'] * len(config['groups'])
    return min(total, config.get('max_group_visits', total))


def reached(value, now):
    return value is not None and now >= value
