import yaml


class PREPConfig(object):
    __slots__ = (
        'names_to_id',
        'types_to_id',
        'allowed_cameras',
	'allowed_ids',
        'integrator',
        'cleaning_method',
        'cleaning_level',
    )

    def __init__(self, config_file):
        with open(config_file) as config:
            config = yaml.load(config)
        for x in self.__slots__:
            if config.get(x) is not None:
                self.__setattr__(x, config.get(x))
            else:
                raise MissingConfigEntry('PREPConfig.__init__()',
                                         'Missing entry in Config file for: '
                                         + x)


class MissingConfigEntry(Exception):
    """Exception raised for missing entries in the given config file.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message
