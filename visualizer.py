from PIL import Image, ImageDraw, ImageFont
import json
from gym_minigrid.minigrid import IDX_TO_COLOR as agent_color

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def add_text(pil_img, right, top, color, text, size=13):
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype("arial.ttf", size)
    draw.text((right, top), text, color, font=font, align="left")
    return pil_img

def add_message_to_image(image, messages):
    new_image = add_margin(image, 0, 0, 100, 0, "black")
    for i, message in enumerate(messages):
        new_image = add_text(new_image, 10, image.height + 10 + 20 * i, agent_color[i], message)
    return new_image

def add_message_to_gif(file_name:str):

    with open(f'./result/{file_name}/log.json') as f:
        data = json.load(f)
    image = Image.open(f'./result/{file_name}/tmp.gif')
    steps = data[0]['steps']
    new_images = []
    for index in range(image.n_frames):
        image.seek(index)
        new_image = add_message_to_image(image, steps[index]["info"]["sended_messages"])
        new_images.append(new_image)

    new_images[0].save(
        f'./result/{file_name}/message.gif', 
        save_all=True, 
        append_images=new_images[1:],
        duration=500,
        loop=0,
    )

add_message_to_gif("message_llm/20240715202640")