import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
        #rgb = self.env.render_no_highlight()

        img = Image.fromarray(np.array(rgb).astype(np.uint8))
        rgb = np.array(img)

        self.rgb_data.append(rgb)
    
    def add_text_to_image(self, pil_img, right, top, color, text, size=13):
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.truetype("arial.ttf", size)
        draw.text((right, top), text, color, font=font, align="left")

    def add_step_to_image(self, image, step):
        self.add_text_to_image(image, 3, 3, "white", f"t={step}")

    def make(self, name = "tmp"):
        images = []
        for i, rgb in enumerate(self.rgb_data):
            rgb = np.array(rgb)
            image = Image.fromarray(rgb.astype('uint8')).convert('RGB')
            self.add_step_to_image(image, i)
            images.append(image)
        images[0].save(f'{self.path}{name}.gif', save_all=True, append_images=images[1:],optimize=False, duration=100, loop=1)
    
    def make_last_frame(self, name = "tmp"):
        rgb = np.array(self.rgb_data[-1])
        image = Image.fromarray(rgb.astype('uint8')).convert('RGB')
        self.add_step_to_image(image, len(self.rgb_data)-1)
        image.save(f'{self.path}{name}.png')