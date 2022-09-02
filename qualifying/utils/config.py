import yaml


def read_yaml_config(path: str):
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config
