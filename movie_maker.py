import numpy as np
from PIL import Image
import visualizer

class MovieMaker:
    def __init__(self, env, path = "./"):
        self.env = env
        self.rgb_data = []
        self.path = path
    
    def reset(self):
        self.rgb_data = []

    def render(self):
        #rgb = self.env.render(mode='rgb_array')
        rgb = self.env.render()

        # img = Image.fromarray(np.array(rgb).astype(np.uint8))
        # if 'messages' in option:
        #     img = visualizer.add_message_to_image(img, option['messages'])
        # rgb = np.array(img)

        self.rgb_data.append(rgb)
    
    def make(self, name = "tmp"):
        images = []
        for rgb in self.rgb_data:
            rgb = np.array(rgb)
            image = Image.fromarray(rgb.astype('uint8')).convert('RGB')
            images.append(image)
        images[0].save(f'{self.path}{name}.gif', save_all=True, append_images=images[1:],optimize=False, duration=100, loop=1)
        
            