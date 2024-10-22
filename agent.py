from gym import Env

class Agent:

    algorithm_name : str

    name : str
    env : Env

    def __init__(self, env):
        self.env = env

    def get_status(self, observation):
        return None
    
    def get_action(self, observation):
        return self.env.action_space.sample()
    
    def train(self, observation, action, reward, next_observation):
        return None
    
    def load_model(self, name:str):
        return None

    def save_model(self, name:str):
        return None
