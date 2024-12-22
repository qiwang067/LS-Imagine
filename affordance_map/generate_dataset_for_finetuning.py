import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import random
import hydra
from omegaconf import OmegaConf
from mineclip import MineCLIP
from concentration import Concentration

def get_next_index(dataset_type, save_path):
    image_files = save_path.joinpath(f'{dataset_type}/images').glob(f'{dataset_type}_image_*.png')
    indices = [int(file.stem.split('_')[-1]) for file in image_files]
    next_index = 1 if not indices else max(indices) + 1
    return next_index

def get_starting_indices(save_path):
    starting_indices = {}
    for dataset_type in ['train', 'val']:
        starting_indices[dataset_type] = get_next_index(dataset_type, save_path)
    return starting_indices

def resize_and_crop_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    target_aspect_ratio = 256 / 160
    new_width, new_height = width, height
    if width / height > target_aspect_ratio:
        new_width = int(target_aspect_ratio * height)
    else:
        new_height = int(width / target_aspect_ratio)
    start_x = (width - new_width) // 2
    start_y = (height - new_height) // 2
    cropped_img = img[start_y:start_y + new_height, start_x:start_x + new_width]
    resized_img = cv2.resize(cropped_img, (256, 160))
    return resized_img

def save_image_and_mask(image, mask, prompt, index, dataset_type, save_path):
    img_save_path = save_path.joinpath(f'{dataset_type}/images')
    mask_save_path = save_path.joinpath(f'{dataset_type}/masks')
    prompt_save_path = save_path.joinpath(f'{dataset_type}/prompts')
    img_save_path.mkdir(parents=True, exist_ok=True)
    mask_save_path.mkdir(parents=True, exist_ok=True)
    prompt_save_path.mkdir(parents=True, exist_ok=True)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_save_path = img_save_path.joinpath(f'{dataset_type}_image_{index:05d}.png')
    cv2.imwrite(str(image_save_path), image)
    
    mask_save_path = mask_save_path.joinpath(f'{dataset_type}_mask_{index:05d}.png')
    mask = (mask * 255).astype(np.uint8)
    cv2.imwrite(str(mask_save_path), mask)

    prompt_save_path = prompt_save_path.joinpath(f'{dataset_type}_prompt_{index:05d}.txt')
    with open(str(prompt_save_path), 'w') as file:
        file.write(prompt)

def generate_triplet(clip_model, device, image, prompt, ker_size, strides):
    concentrator = Concentration(clip_model, device, image, prompt, ker_size, strides)

    concentrator.generate_masks() 
    mask = concentrator.merge_masks()  
    
    mask = 1 / (1 + np.exp(-1.2 * (mask - 21.8)))
    
    return image, mask, prompt

def process_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((256, 160))
    return np.array(image)

def process_mask(mask_path):
    mask = Image.open(mask_path).convert('L')
    mask = mask.resize((256, 160))
    return np.expand_dims(np.array(mask) / 255.0, axis=-1)

def process_prompt(prompt_path, clip_model):
    with open(prompt_path, 'r') as file:
        prompt = file.read().strip()

        pmt = clip_model.encode_text([prompt])
        pmt = pmt.squeeze(0).cpu().detach().numpy()
        return pmt
    
def generate_data_pairs(base_dir, output_dir, list_file, clip_model):
    os.makedirs(output_dir, exist_ok=True)

    image_dir = os.path.join(base_dir, 'images')
    mask_dir = os.path.join(base_dir, 'masks')
    prompt_dir = os.path.join(base_dir, 'prompts')

    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))
    prompt_files = sorted(os.listdir(prompt_dir))

    npz_names = []

    for index, (image_file, mask_file, prompt_file) in enumerate(zip(image_files, mask_files, prompt_files), start=1):
        npz_name = f'train_{index:05d}.npz'
        image_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, mask_file)
        prompt_path = os.path.join(prompt_dir, prompt_file)

        image = process_image(image_path)
        mask = process_mask(mask_path)
        prompt = process_prompt(prompt_path, clip_model)

        # Save to .npz file
        npz_path = os.path.join(output_dir, npz_name)
        np.savez(npz_path, image=image, label=mask, prompt=prompt)

        # Add the name to the list
        npz_names.append(npz_name[:-4])

    os.makedirs(os.path.dirname(list_file), exist_ok=True)

    # Write all npz names to a list file
    with open(list_file, 'w') as file:
        file.write("\n".join(npz_names))

@hydra.main(config_name="conf", config_path=".", version_base="1.1")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OmegaConf.set_struct(cfg, False)
    ker_size = cfg.pop("ker_size")
    strides = cfg.pop("strides")
    finetune_task = cfg.pop("finetune_task")
    prompt = cfg.pop("prompt")
    ckpt = cfg.pop("ckpt")
    OmegaConf.set_struct(cfg, True)

    clip_model = MineCLIP(**cfg).to(device)
    clip_model.load_ckpt(ckpt.path, strict=True)

    root_path = Path(f'affordance_map/finetune_unet/rollouts/{finetune_task}')
    image_paths = list(root_path.rglob('*.png'))
    random.shuffle(image_paths)

    print("image_num:", len(image_paths))

    save_path = Path(f'affordance_map/datasets/triplets/{finetune_task}')
    starting_indices = get_starting_indices(save_path)

    for image_path in tqdm(image_paths, desc=f'Processing images'):
        image = resize_and_crop_image(str(image_path))
        dataset_type = 'train'
        index = starting_indices[dataset_type]
        starting_indices[dataset_type] += 1
        image, mask, prompt = generate_triplet(clip_model, device, image, prompt, ker_size, strides)
        save_image_and_mask(image, mask, prompt, index, dataset_type, save_path)

    base_dir = f'affordance_map/datasets/triplets/{finetune_task}/train'
    output_dir = f'affordance_map/datasets/Minecraft/{finetune_task}/train_npz'
    list_file = f'affordance_map/lists/lists_Minecraft/{finetune_task}/train.txt'
    generate_data_pairs(base_dir, output_dir, list_file, clip_model)


if __name__ == '__main__':
    main()