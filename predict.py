# inference.py


import copy
import json
import base64
import logging
import argparse
import torch
import os
from io import BytesIO
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from qwen2vl import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from transformers import AutoModel, AutoTokenizer,AutoProcessor
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import requests
import transformers
import numpy as np
import random
from tqdm import tqdm
from datasets import load_dataset,Dataset

logger = logging.getLogger(__name__)
os.environ["WANDB_DISABLED"] = "true"
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_GEN_KWARGS = dict(
    num_beams=1,
    max_new_tokens=1024,
    do_sample=False,
)


def build_transform_internvl2_5(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img), T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
    return transform

def find_closest_aspect_ratio_internvl2_5(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess_internvl2_5(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio_internvl2_5(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size, ((i % (target_width // image_size)) + 1) * image_size, ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image_internvl2_5(image, input_size=448, max_num=6):
    transform = build_transform_internvl2_5(input_size=input_size)
    images = dynamic_preprocess_internvl2_5(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)   # [num_patches, 3, 448, 448]
    return pixel_values

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    
def parse_args():
    parser = argparse.ArgumentParser(description="A simple inference script to test token prune and kv cache compression methods.")
    # settings for path/basic
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--image_path", type=str, default="/home/lyli/EffiVLM-Bench-main/test/llava_v1_5_radar.jpg")
    parser.add_argument("--question", type=str, default="What is shown in this image?",help="the question to ask the model")

    # settings for model configuration
    parser.add_argument('--pretrained', type=str, default="/home/lyli/models/Qwen2-VL-7B-Instruct", help='Pretrained model path or identifier.')
    parser.add_argument('--model_name', type=str, choices=['llava-onevision-qwen2',
                                                       'qwen2-vl', 
                                                       'internvl2_5'], help='Model name. such as llava-onevision-qwen2-7b-ov , Qwen2-VL-7B-Instruct , InternVL2_5-4B , InternVL2_5-38B.')
    parser.add_argument("--use_cache", type=bool, default=True, help="")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="")
    parser.add_argument("--temperature", type=float, default=0, help="")
    parser.add_argument("--top_p", type=float, default=None, help="")
    parser.add_argument("--num_beams", type=int, default=1, help="")
    parser.add_argument("--attn_implementation", type=str, default="eager", help="")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", help="")
    parser.add_argument("--multimodal", type=bool, default=True, help="")
    parser.add_argument("--device", type=str, default="cuda:0", help="")
    parser.add_argument("--device_map", type=str, default="cuda:0", help="")
    # for qwen2-vl
    parser.add_argument("--max_pixels", type=int, default=1000*1000, help="the max pixels of the image for qwen2-vl")
    parser.add_argument("--min_pixels", type=int, default=1024, help="the min pixels of the image for qwen2-vl")
    
    # settings for kv cache method
    parser.add_argument('--method', type=str, choices=['random', # kv cache method
                                                       'streamingllm', 
                                                       'h2o', 
                                                       'snapkv', 
                                                       'look-m', 
                                                       'vl-cache', 
                                                       'pyramidkv',
                                                       'fastv', # token prune method
                                                       'visionzip',
                                                       'prumerge+'], help='KV cache method to use.')
    parser.add_argument("--merge", type=bool, default=True, help="merge switch of kv merge method(look-m)") # for look-m
    parser.add_argument("--head_adaptive", type=bool, default=True, help='head adaptive of h2o,snapkv,pyramidkv') # for h2o,snapkv,pyramidkv
    parser.add_argument("--pooling", type=str, default="avgpool", help='pooling of snapkv,pyramidkv') # for snapkv,pyramidkv
    parser.add_argument("--layer_adaptive", type=bool, default=True, help='layer adaptive of vl-cache') # for vl-cache
    parser.add_argument("--vlcache_different_window_per_layer", type=bool, default=False, help='vlcache different window per layer of vl-cache') # for vl-cache
    parser.add_argument("--budgets", type=float, default=0.028, help="budgets of Key Cache")

    
    return parser.parse_args()

def replace_layers(args,model):
    from kv_cache_compression.monkeypatch import replace_qwen,replace_qwen2vl,replace_internvl2_5

    if "llava-onevision-qwen2" in args.model_name.lower():
        replace_qwen(args,model,args.method.lower())
    elif "qwen2-vl" in args.model_name.lower():
        replace_qwen2vl(args,model,args.method.lower())
    elif "internvl2_5" in args.model_name.lower():
        replace_internvl2_5(args,model,args.method.lower())
    else:
        raise ValueError(f"Model name {args.model_name} not supported")


def run_inference(args):

    if "llava-onevision" in args.model_name.lower():
        run_inference_ov(args)
    elif "qwen2-vl" in args.model_name.lower():
        # run_inference_qwen2vl(args)
        run_qwen2vl_mme(args)
    elif "internvl2_5" in args.model_name.lower():
        run_inference_internvl2_5(args)
    else:
        raise ValueError(f"Model name {args.model_name} not supported")
    

def load_model_llava_ov(args, pretrained, model_name):
    llava_model_args = {
        "attn_implementation": "eager", # eager for default
        "device_map": args.device_map, 
        "torch_dtype": args.torch_dtype,
        "multimodal": args.multimodal
    }
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, **llava_model_args)
    
    # replace some layers impl of kv cache method
    model.eval()
    return tokenizer, model, image_processor, max_length


def load_model_qwen2vl(args, pretrained, model_name):

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        pretrained,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        attn_implementation="flash_attention_2", # flash_attention_2 for default
    ).eval()
    qwen2vl_processor = AutoProcessor.from_pretrained(pretrained, torch_dtype=torch.float16)
    # qwen2vl_processor = Qwen2VLProcessor.from_pretrained(pretrained, max_pixels=args.max_pixels, min_pixels=args.min_pixels)
    qwen2vl_tokenizer = AutoTokenizer.from_pretrained(pretrained)

    return model, qwen2vl_processor, qwen2vl_tokenizer


def load_model_internvl2_5(args, pretrained, model_name):
        
    model = AutoModel.from_pretrained(pretrained, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, device_map=args.device_map, use_flash_attn=False).eval()
    intern2_5_tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True, device_map=args.device_map)
    return model, intern2_5_tokenizer

def run_inference_ov(args):
    
    tokenizer, model, image_processor, max_length = load_model_llava_ov(args, args.pretrained, args.model_name)
    replace_layers(args,model)

    device = args.device
    image_path = args.image_path
    image = Image.open(image_path)
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.bfloat16, device=device) for _image in image_tensor]
    
    question = DEFAULT_IMAGE_TOKEN + "\n" + args.question
    conv_template = "qwen_1_5" 

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    image_sizes = [image.size]

    cont = model.generate(
        input_ids,  
        images=image_tensor,  
        image_sizes=image_sizes,
        do_sample=True if args.temperature > 0 else False,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        method=args.method
    )
    output = tokenizer.batch_decode(cont, skip_special_tokens=True)
    print(output[0])

def run_qwen2vl_mme(args):
    model, qwen2vl_processor, qwen2vl_tokenizer = load_model_qwen2vl(args, args.pretrained, args.model_name)
    replace_layers(args,model)
    data = load_dataset("lmms-lab/MME")['test']

    print(111)
    category = {}
    res = {}
    for _ in data:
        if _['category'] in category:
            category[_['category']] += 1 
        else:
            category[_['category']] =1
            res[_['category']] = []

    print(category)
    print(data)
    #280,280-> 721
    #224,224-> 695
    #icae -> 731

    #icae ->737
    #560 -> 800
    count = 0
    res_list = []
    data = tqdm(data)
    all_time =0
    all_memory = 0
    for i,d in enumerate(data):
        messages = []
        # processed_visuals = []
        # context = args.question
        message = [{"role": "system", "content": "You are a helpful assistant."}]
        # visual = Image.open(args.image_path)
        visual = d["image"]
        base64_image = visual.convert("RGB")
        buffer = BytesIO()
        base64_image.save(buffer, format="JPEG")
        base64_bytes = base64.b64encode(buffer.getvalue())
        base64_string = base64_bytes.decode("utf-8")

        message.append({"role": "user", "content": [{"type": "image", "image": f"data:image/jpeg;base64,{base64_string}",  "resized_height": 336,
                        "resized_width": 336,}, {"type": "text", "text": d["question"]}]})
        messages.append(message)
        texts = [qwen2vl_processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        print(image_inputs)
        inputs = qwen2vl_processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        print(inputs["pixel_values"].dtype)
        if args.device_map == "auto":
            inputs = inputs.to("cuda")
        else:
            inputs = inputs.to(args.device)
        pad_token_id = qwen2vl_tokenizer.pad_token_id
        # print(inputs)
        # print(inputs["pixel_values"].shape)
        # generate
        import time
        torch.cuda.synchronize()  # 开始前同步一次，保证干净
        start = time.time()
        
        cont = model.generate(
            **inputs,
            eos_token_id= qwen2vl_tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=2,
            use_cache=args.use_cache,
        )
        end = time.time()
        torch.cuda.synchronize()  # 开始前同步一次，保证干净
        print(f"generate time:{end-start}")
        all_time +=end-start
        print(f"avg time:{all_time/(i+1)}")
        peak_mem_bytes = torch.cuda.max_memory_allocated("cuda")
        peak_mem_MB = peak_mem_bytes / 1024 / 1024
        print(f"推理过程中最高显存占用: {peak_mem_MB:.2f} MB")
        all_memory += peak_mem_MB
        print(f"avg memory:{all_memory/(i+1)}")
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        answers = qwen2vl_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # print answer
        print(answers[0])
        output_text = answers
        print(f"out:{output_text[0]}")
        output_text[0] = output_text[0].lower()
        d['answer'] = d['answer'].lower()
        print(f"gt:{d['answer']}")
        print(count)
        if output_text[0]==d['answer'] or output_text[0] in d['answer'] or d['answer'] in output_text[0]:
            res[d['category']].append(True)
            res_list.append(True)
            count+=1
        else:
            res_list.append(False)
            res[d['category']].append(False)


    #acc   
    print(f"acc:{sum(res_list)}:{len(res_list)}")


    #acc+
    cc = 0
    for i in range(0,len(res_list),2):
        if res_list[i]==True and res_list[i+1]==True:
            cc+=1    
            
    print(f"acc++:{cc}:{len(res_list)}")

    score = 0
    for k,v in category.items():
        acc = sum(res[k])
        acc_plus = 0
        for i in range(0,v,2):
            if res[k][i]==True and res[k][i+1]==True:
                acc_plus+=1
        print(f"{k}: acc{100*acc/v}   acc+{100*acc_plus/(v/2)}")
        score+=100*acc/v + 100*acc_plus/(v/2)

    print(score)


    score = 0
    for k,v in category.items():
            acc = sum(res[k])
            acc_plus = 0
            for i in range(0,v,2):
                if res[k][i]==True and res[k][i+1]==True:
                    acc_plus+=1
            print(f"{k}: acc{100*acc/v}   acc+{100*acc_plus/(v/2)}")
            if k in ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"]:
                score+=100*acc/v + 100*acc_plus/(v/2)
    print(score)

import io

def run_qwen2vl_mmb(args):
    model, qwen2vl_processor, qwen2vl_tokenizer = load_model_qwen2vl(args, args.pretrained, args.model_name)
    replace_layers(args,model)
    data = load_dataset("lmms-lab/MME")['test']

    print(111)
    category = {}
    res = {}

    count = 0
    res_list = []
    data = tqdm(data)
    data = load_dataset("csv",data_files="/home/ypzheng/MMBench_DEV_EN.tsv",delimiter="\t")['train']
    res_df  = data.to_pandas()

    for d in data:
        messages = []
        # processed_visuals = []
        # context = args.question
        message = [{"role": "system", "content": "You are a helpful assistant."}]
        # visual = Image.open(args.image_path)
        if d["image"].isdigit():
            image = Image.open(os.path.join("/home/ypzheng/LMUData/images/MMBench", d["image"])+".jpg")
        else:
            img_data = base64.b64decode(d["image"])
            image_bytes = io.BytesIO(img_data)
            # 打开字节流，转换为 PIL 图像
            image = Image.open(image_bytes)
        visual = image
        base64_image = visual.convert("RGB")
        buffer = BytesIO()
        base64_image.save(buffer, format="JPEG")
        base64_bytes = base64.b64encode(buffer.getvalue())
        base64_string = base64_bytes.decode("utf-8")

        message.append({"role": "user", "content": [{"type": "image", "image": f"data:image/jpeg;base64,{base64_string}",  "resized_height": 336,
                        "resized_width": 336,}, {"type": "text", "text":"You must choose among A,B,C and D. The question is: " + d["question"] + " "
                            + ("A: " + d["A"] if d["A"] else "A: None") + " "
                            + ("B: " + d["B"] if d["B"] else "B: None") + " "
                            + ("C: " + d["C"] if d["C"] else "C: None") + " "
                            + ("D: " + d["D"] if d["D"] else "D: None") + " "
                            + " Please just answer A B C or D. Don't answer any other words."}]})
        messages.append(message)
        texts = [qwen2vl_processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        print(image_inputs)
        inputs = qwen2vl_processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

        if args.device_map == "auto":
            inputs = inputs.to("cuda")
        else:
            inputs = inputs.to(args.device)
        pad_token_id = qwen2vl_tokenizer.pad_token_id
        # print(inputs)
        # print(inputs["pixel_values"].shape)
        # generate
        cont = model.generate(
            **inputs,
            eos_token_id= qwen2vl_tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=args.use_cache,
        )

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        answers = qwen2vl_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # print answer
        output_text = answers
        print(f"out:{output_text[0]}")
        res_list.append(output_text[0])


    
    res_df["prediction"] = res_list

    res_df.to_excel("prediction_4token_fastv.xlsx",index=False)
   
def run_qwen2vl_mmb_cn(args):
    model, qwen2vl_processor, qwen2vl_tokenizer = load_model_qwen2vl(args, args.pretrained, args.model_name)
    replace_layers(args,model)
    data = load_dataset("lmms-lab/MME")['test']

    print(111)
    category = {}
    res = {}

    count = 0
    res_list = []
    data = tqdm(data)
    data = load_dataset("csv",data_files="/home/ypzheng/mmbench_dev_cn_20231003.tsv",delimiter="\t")['train']
    res_df  = data.to_pandas()

    for d in data:
        messages = []
        # processed_visuals = []
        # context = args.question
        message = [{"role": "system", "content": "You are a helpful assistant."}]
        # visual = Image.open(args.image_path)
        if d["image"].isdigit():
            image = Image.open(os.path.join("/home/ypzheng/LMUData/images/MMBench", d["image"])+".jpg")
        else:
            img_data = base64.b64decode(d["image"])
            image_bytes = io.BytesIO(img_data)
            # 打开字节流，转换为 PIL 图像
            image = Image.open(image_bytes)
        visual = image
        base64_image = visual.convert("RGB")
        buffer = BytesIO()
        base64_image.save(buffer, format="JPEG")
        base64_bytes = base64.b64encode(buffer.getvalue())
        base64_string = base64_bytes.decode("utf-8")

        message.append({"role": "user", "content": [{"type": "image", "image": f"data:image/jpeg;base64,{base64_string}",  "resized_height": 336,
                        "resized_width": 336,}, {"type": "text", "text":"The question is: " + d["question"] + " "
                            + ("A: " + d["A"] if d["A"] else "A: None") + " "
                            + ("B: " + d["B"] if d["B"] else "B: None") + " "
                            + ("C: " + d["C"] if d["C"] else "C: None") + " "
                            + ("D: " + d["D"] if d["D"] else "D: None") + " "
                            + " 请直接回答选项字母。"}]})
        messages.append(message)
        texts = [qwen2vl_processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        print(image_inputs)
        inputs = qwen2vl_processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

        if args.device_map == "auto":
            inputs = inputs.to("cuda")
        else:
            inputs = inputs.to(args.device)
        pad_token_id = qwen2vl_tokenizer.pad_token_id
        # print(inputs)
        # print(inputs["pixel_values"].shape)
        # generate
        cont = model.generate(
            **inputs,
            eos_token_id= qwen2vl_tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=args.use_cache,
        )

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        answers = qwen2vl_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # print answer
        output_text = answers
        print(f"out:{output_text[0]}")
        res_list.append(output_text[0])


    
    res_df["prediction"] = res_list

    res_df.to_excel("prediction_cn.xlsx",index=False)



def run_qwen2vl_pope(args):
    model, qwen2vl_processor, qwen2vl_tokenizer = load_model_qwen2vl(args, args.pretrained, args.model_name)
    replace_layers(args,model)
    data = load_dataset("lmms-lab/POPE")["test"]


    count = 0
    
    TP =0
    FP=0
    TN=0
    FN=0    
    all_time =0
    all_memory = 0
    for i,d in enumerate(data):
        import time
        
        messages = []
        # processed_visuals = []
        # context = args.question
        message = [{"role": "system", "content": "You are a helpful assistant."}]
        # visual = Image.open(args.image_path)
        # if d["image"].isdigit():
        #     image = Image.open(os.path.join("/home/ypzheng/LMUData/images/MMBench", d["image"])+".jpg")
        # else:
        #     img_data = base64.b64decode(d["image"])
        #     image_bytes = io.BytesIO(img_data)
        #     # 打开字节流，转换为 PIL 图像
        #     image = Image.open(image_bytes)
        visual = d["image"]
        base64_image = visual.convert("RGB")
        buffer = BytesIO()
        base64_image.save(buffer, format="JPEG")
        base64_bytes = base64.b64encode(buffer.getvalue())
        base64_string = base64_bytes.decode("utf-8")

        message.append({"role": "user", "content": [{"type": "image", "image": f"data:image/jpeg;base64,{base64_string}",  }, {"type": "text", "text": d['question']+" please answer yes or no."}]})
        messages.append(message)
        texts = [qwen2vl_processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        print(image_inputs)
        inputs = qwen2vl_processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

        if args.device_map == "auto":
            inputs = inputs.to("cuda")
        else:
            inputs = inputs.to(args.device)
        pad_token_id = qwen2vl_tokenizer.pad_token_id
        # print(inputs)
        # print(inputs["pixel_values"].shape)
        # generate
        
        print(inputs["input_ids"].shape)
        torch.cuda.synchronize()  # 开始前同步一次，保证干净
        start = time.time()
        
        cont = model.generate(
            **inputs,
            eos_token_id= qwen2vl_tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=2,
            use_cache=args.use_cache,
        )
        end = time.time()
        torch.cuda.synchronize()  # 开始前同步一次，保证干净
        print(f"generate time:{end-start}")
        all_time +=end-start
        print(f"avg time:{all_time/(i+1)}")
        peak_mem_bytes = torch.cuda.max_memory_allocated("cuda")
        peak_mem_MB = peak_mem_bytes / 1024 / 1024
        print(f"推理过程中最高显存占用: {peak_mem_MB:.2f} MB")
        all_memory += peak_mem_MB
        print(f"avg memory:{all_memory/(i+1)}")

        
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        answers = qwen2vl_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # print answer
        output_text = answers
        print(f"out:{output_text[0]}")
        pred = output_text[0].lower()
        label = d['answer'].lower()
        
        if (pred == "yes" or "yes" in pred) and label == "yes":
            TP += 1
        elif(pred == "yes" or "yes" in pred)  and label == "no":
            FP += 1
        elif (pred == "no" or "no" in pred)  and label == "no":
            TN += 1
        elif (pred == "no" or "no" in pred)  and label == "yes":
            FN += 1
        else:
            print(123)
        
        if pred==label:
            count+=1
        print(count)
        
    print(count)

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    # print('Yes ratio: {}'.format(yes_ratio))


def run_qwen2vl_mmvet(args):
    model, qwen2vl_processor, qwen2vl_tokenizer = load_model_qwen2vl(args, args.pretrained, args.model_name)
    replace_layers(args,model)
    data = load_dataset("lmms-lab/MMVet")['test']

    res_dict = {}

    for d in data:
        messages = []
        # processed_visuals = []
        # context = args.question
        message = [{"role": "system", "content": "You are a helpful assistant."}]
        # visual = Image.open(args.image_path)
        # if d["image"].isdigit():
        #     image = Image.open(os.path.join("/home/ypzheng/LMUData/images/MMBench", d["image"])+".jpg")
        # else:
        #     img_data = base64.b64decode(d["image"])
        #     image_bytes = io.BytesIO(img_data)
        #     # 打开字节流，转换为 PIL 图像
        #     image = Image.open(image_bytes)
        visual = d["image"]
        base64_image = visual.convert("RGB")
        buffer = BytesIO()
        base64_image.save(buffer, format="JPEG")
        base64_bytes = base64.b64encode(buffer.getvalue())
        base64_string = base64_bytes.decode("utf-8")

        message.append({"role": "user", "content": [{"type": "image", "image": f"data:image/jpeg;base64,{base64_string}",  "resized_height": 336,
                        "resized_width": 336,}, {"type": "text", "text": d['question']}]})
        messages.append(message)
        texts = [qwen2vl_processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        print(image_inputs)
        inputs = qwen2vl_processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

        if args.device_map == "auto":
            inputs = inputs.to("cuda")
        else:
            inputs = inputs.to(args.device)
        pad_token_id = qwen2vl_tokenizer.pad_token_id
        # print(inputs)
        # print(inputs["pixel_values"].shape)
        # generate
        cont = model.generate(
            **inputs,
            eos_token_id= qwen2vl_tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=args.use_cache,
        )

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        answers = qwen2vl_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # print answer
        output_text = answers
        print(f"out:{output_text[0]}")
        res_dict[f"{d['question_id']}"] = output_text[0]

        # print(count)

    with open("mmvet_8x.json","w+") as f:
        json.dump(res_dict,f)
    # print(count)

def run_qwen2vl_case(args):
    model, qwen2vl_processor, qwen2vl_tokenizer = load_model_qwen2vl(args, args.pretrained, args.model_name)
    replace_layers(args,model)
    data = load_dataset("json",data_files='/home/ypzheng/VLM_ICAE/data/case/data.json')['train']


    res_dict = {}
    count = 0
    for i,d in enumerate(data):
        messages = []
        # processed_visuals = []
        # context = args.question
        message = [{"role": "system", "content": "You are a helpful assistant."}]
        # visual = Image.open(args.image_path)
        # if d["image"].isdigit():
        image = Image.open(d["image"])
        # else:
        #     img_data = base64.b64decode(d["image"])
        #     image_bytes = io.BytesIO(img_data)
        #     # 打开字节流，转换为 PIL 图像
        #     image = Image.open(image_bytes)
        visual = image
        base64_image = visual.convert("RGB")
        buffer = BytesIO()
        base64_image.save(buffer, format="JPEG")
        base64_bytes = base64.b64encode(buffer.getvalue())
        base64_string = base64_bytes.decode("utf-8")
# 
                       
        message.append({"role": "user", "content": [{"type": "image", "image": f"data:image/jpeg;base64,{base64_string}", "resized_height": 336,  "resized_width": 336, }, {"type": "text", "text": d['question']}]})
        messages.append(message)
        texts = [qwen2vl_processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        print(image_inputs)
        inputs = qwen2vl_processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

        if args.device_map == "auto":
            inputs = inputs.to("cuda")
        else:
            inputs = inputs.to(args.device)
        pad_token_id = qwen2vl_tokenizer.pad_token_id
        # print(inputs)
        # print(inputs["pixel_values"].shape)
        # generate
        cont = model.generate(
            **inputs,
            eos_token_id= qwen2vl_tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=512,
            use_cache=args.use_cache,
        )

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        answer = qwen2vl_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # print answer
        output_text = answer
        print(f"out:{output_text[0]}")
        # generated_text = output_text[0].strip().lower()
    

def run_qwen2vl_gqa(args):
    model, qwen2vl_processor, qwen2vl_tokenizer = load_model_qwen2vl(args, args.pretrained, args.model_name)
    replace_layers(args,model)
    data = load_dataset("json",data_files='/home/ypzheng/llava_gqa_testdev_balanced.jsonl')['train']
    with open("/home/ypzheng/testdev_balanced_questions.json","r") as f:
        answers = json.load(f)

    res_dict = {}
    count = 0
    for i,d in enumerate(data):
        messages = []
        # processed_visuals = []
        # context = args.question
        message = [{"role": "system", "content": "You are a helpful assistant."}]
        # visual = Image.open(args.image_path)
        # if d["image"].isdigit():
        image = Image.open(os.path.join("/home/ypzheng/VLM_ICAE/data/gqa_testdev_images", d["image"]))
        # else:
        #     img_data = base64.b64decode(d["image"])
        #     image_bytes = io.BytesIO(img_data)
        #     # 打开字节流，转换为 PIL 图像
        #     image = Image.open(image_bytes)
        visual = image
        base64_image = visual.convert("RGB")
        buffer = BytesIO()
        base64_image.save(buffer, format="JPEG")
        base64_bytes = base64.b64encode(buffer.getvalue())
        base64_string = base64_bytes.decode("utf-8")
# 
                       
        message.append({"role": "user", "content": [{"type": "image", "image": f"data:image/jpeg;base64,{base64_string}", "resized_height": 336,  "resized_width": 336, }, {"type": "text", "text": d['text']}]})
        messages.append(message)
        texts = [qwen2vl_processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        print(image_inputs)
        inputs = qwen2vl_processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

        if args.device_map == "auto":
            inputs = inputs.to("cuda")
        else:
            inputs = inputs.to(args.device)
        pad_token_id = qwen2vl_tokenizer.pad_token_id
        # print(inputs)
        # print(inputs["pixel_values"].shape)
        # generate
        cont = model.generate(
            **inputs,
            eos_token_id= qwen2vl_tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=10,
            use_cache=args.use_cache,
        )

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        answer = qwen2vl_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # print answer
        output_text = answer
        print(f"out:{output_text[0]}")
        generated_text = output_text[0].strip().lower()
        # print(answers[d['question_id']])
        gt_answer = answers[d['question_id']]["answer"].strip().lower()
        if generated_text == gt_answer or generated_text in gt_answer or gt_answer in generated_text:
            count+=1
        print(count)
        print(count/(i+1))
      

        # print(count)


    # print(count)


def run_inference_qwen2vl(args):

    model, qwen2vl_processor, qwen2vl_tokenizer = load_model_qwen2vl(args, args.pretrained, args.model_name)
    replace_layers(args,model)

    messages = []
    processed_visuals = []
    context = args.question
    message = [{"role": "system", "content": "You are a helpful assistant."}]
    # process image
    visual = Image.open(args.image_path)
    base64_image = visual.convert("RGB")
    buffer = BytesIO()
    base64_image.save(buffer, format="JPEG")
    base64_bytes = base64.b64encode(buffer.getvalue())
    base64_string = base64_bytes.decode("utf-8")

    message.append({"role": "user", "content": [{"type": "image", "image": f"data:image/jpeg;base64,{base64_string}", "max_pixels": args.max_pixels, "min_pixels": args.min_pixels}, {"type": "text", "text": context}]})
    messages.append(message)
    texts = [qwen2vl_processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    print(image_inputs)
    inputs = qwen2vl_processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

    if args.device_map == "auto":
        inputs = inputs.to("cuda")
    else:
        inputs = inputs.to(args.device)
    pad_token_id = qwen2vl_tokenizer.pad_token_id
    print(inputs)
    print(inputs["pixel_values"].shape)
    # generate
    cont = model.generate(
        **inputs,
        eos_token_id= qwen2vl_tokenizer.eos_token_id,
        pad_token_id=pad_token_id,
        do_sample=True if args.temperature > 0 else False,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        use_cache=args.use_cache,
    )

    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
    answers = qwen2vl_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # print answer
    print(answers[0])


def run_inference_internvl2_5(args):
    model, intern2_5_tokenizer = load_model_internvl2_5(args, args.pretrained, args.model_name)
    # process image
    replace_layers(args,model)
    image = Image.open(args.image_path)
    visuals = [load_image_internvl2_5(image).to(torch.bfloat16).cuda()]
    pixel_values = torch.cat(visuals, dim=0)
    num_patches_list = [visual.size(0) for visual in visuals]
    # get prompt
    image_tokens = ["<image>"] * len(visuals)
    image_tokens = " ".join(image_tokens)
    contexts = image_tokens + "\n" + args.question
    # generate
    gen_kwargs = dict(
        do_sample=True if args.temperature > 0 else False,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
    )
    response, history = model.chat(intern2_5_tokenizer, pixel_values, contexts, gen_kwargs, num_patches_list=num_patches_list, history=None, return_history=True)
    print(response)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)