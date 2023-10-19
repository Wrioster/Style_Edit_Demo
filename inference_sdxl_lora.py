from diffusers import StableDiffusionXLPipeline
import torch
import torch.nn as nn
from torch.nn import Parameter
import os
import os.path as osp
from diffusers import PromptRefiner


config_dict = {
    #"objects": ["baby penguin", "robot", "chair",  "flower", "pine","mountain"],  # test prompt
    "objects": ["coconut tree", "tree", "dog", "apple", "shoe", "pear", "water droplet"],
    "train_prompt":"A coconut tree in flat color style", #EDIT
    "style": "flat color style", #EDIT
    "iters": [1000],
    "repeats": 3,
    "lora_path": "saved_weights/flat_color_mode0", #EDIT
    "PRF_mode": 1, #EDIT THIS!
    "use_neg_prompt": 1, #EDIT (1, use; 0, not use)
    "show_prompts": 1,
    "gpu_id": 0,
    "num_inference_steps": 30,
    "cfg": 7.5
}

def get_img_save_path():
    path0 = 'outcomes/generation'
    path_style = ""
    train_prompt_list = config_dict["train_prompt"].split(" ")
    ids_1 = train_prompt_list.index("in")
    ids_2 = len(train_prompt_list)
    for p in train_prompt_list[ids_1+1:ids_2-1]:
        path_style += p
        path_style += "_"
    path0 = os.path.join(path0, path_style[:-1])
    if not os.path.exists(path0):
        os.mkdir(path0)
    
    style_list = config_dict["style"].split(" ")
    style_edit = ""
    for p in style_list[:-1]:
        style_edit += p
        style_edit += "_"
    if style_edit == path_style:
        #without style edit
        path0 = os.path.join(path0, "WithoutStyleEdit")
    else:
        path0 = os.path.join(path0, style_edit[:-1])
    if not os.path.exists(path0):
        os.mkdir(path0)
        
    lora_path = config_dict["lora_path"].split("_")
    mode = "Neg_" + lora_path[-1] if config_dict["use_neg_prompt"] == 1 else lora_path[-1]
    path0 = os.path.join(path0, mode)
    if not os.path.exists(path0):
        os.mkdir(path0)
    
    return path0
    

def get_neg_prompt(prompt, train_prompt, conjunction = ", "):
    train_prompt_list = train_prompt.split(" ")
    prompt_list = prompt.split(" ")
    neg_prompt = ""
    for p in train_prompt_list:
        if p not in prompt_list:
            neg_prompt += p
            neg_prompt += conjunction
    len_conjunction = len(conjunction)
    return neg_prompt[:-len_conjunction]

def show_prompts(prompts, neg_prompts):
    for id, (p, np) in enumerate(zip(prompts, neg_prompts)):
        print("-"*50)
        print(f"Case:{id}")
        print(f"pos: {p}")
        print(f"neg: {np}")
    print("-"*50)

# load model
pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
pipe.to("cuda:{}".format(config_dict["gpu_id"]))
lora_path = config_dict["lora_path"]

# prompt as "A {object} in {style}"
objects = config_dict["objects"]
style = config_dict["style"]
prompts = [
    f"A {object} in {style}" for object in objects
]

neg_prompts = [get_neg_prompt(m, config_dict["train_prompt"]) for m in prompts]

if config_dict["show_prompts"]==1:
    show_prompts(prompts, neg_prompts)

# save images dir
root_dir = get_img_save_path()
if not osp.isdir(root_dir):
    os.mkdir(root_dir)
paths = ["{}/{}".format(root_dir, object.replace(" ", "_")) for object in objects]
for path in paths:
    if not osp.isdir(path):
        os.mkdir(path)
id2path = {i:p for i, p in enumerate(paths)}

mode = config_dict["PRF_mode"]
iters = config_dict["iters"]
for iter in iters:
    print(f"generating iter{iter}'s images ...")
    model_path = f"{lora_path}/checkpoint-{iter}"  # number of epoch == number of iter
    pipe.set_PRF_mode(mode)
    pipe.set_text_encoders(lora_path, f"checkpoint-{iter}")
    pipe.load_lora_weights(model_path)
    
    for r in range(config_dict["repeats"]):
        if config_dict["use_neg_prompt"] == 0:
            for id, prompt in enumerate(prompts):
                image = pipe(
                    prompt=prompt,
                    num_inference_steps=config_dict["num_inference_steps"], 
                    guidance_scale=config_dict["cfg"]).images[0]
                image.save(f"{id2path[id]}/iter{iter}_{r}.png")
        else:
            for id, (prompt, neg_prompt) in enumerate(zip(prompts, neg_prompts)):
                image = pipe(
                    prompt=prompt,
                    negative_prompt = neg_prompt,
                    num_inference_steps=config_dict["num_inference_steps"], 
                    guidance_scale=config_dict["cfg"]).images[0]
                image.save(f"{id2path[id]}/iter{iter}_{r}.png")