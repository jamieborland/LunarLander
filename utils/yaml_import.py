import yaml

def load_config(agent_name):
    with open(f"configs/{agent_name}.yaml", "r") as f:
        return yaml.safe_load(f) or {}