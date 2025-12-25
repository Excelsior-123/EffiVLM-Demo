import streamlit as st
import sys
import os
import time
import pandas as pd
import torch
from PIL import Image
import argparse
from io import BytesIO
import base64
import copy
import logging
import numpy as np
import random
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import gc

# Set environment variables
os.environ["WANDB_DISABLED"] = "true"
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from qwen2vl import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from transformers import AutoModel, AutoTokenizer
from kv_cache_compression.monkeypatch import replace_qwen, replace_qwen2vl, replace_internvl2_5

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Helper functions from predict.py
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

# Mock Args class
class Args:
    def __init__(self, **kwargs):
        self.seed = 42
        self.image_path = None # Will be handled separately
        self.question = "What is shown in this image?"
        self.pretrained = "/home/lyli/models/Qwen2-VL-7B-Instruct"
        self.model_name = "qwen2-vl"
        self.use_cache = True
        self.max_new_tokens = 2048
        self.temperature = 0
        self.top_p = None
        self.num_beams = 1
        self.attn_implementation = "eager"
        self.torch_dtype = "bfloat16"
        self.multimodal = True
        self.device = "cuda:0"
        self.device_map = "cuda:0"
        self.max_pixels = 1024
        self.min_pixels = 1024
        self.method = "random"
        self.merge = True
        self.head_adaptive = True
        self.pooling = "avgpool"
        self.layer_adaptive = True
        self.vlcache_different_window_per_layer = False
        self.budgets = 0.4
        
        for k, v in kwargs.items():
            setattr(self, k, v)

def replace_layers(args, model):
    if "llava-onevision-qwen2" in args.model_name.lower():
        replace_qwen(args, model, args.method.lower())
    elif "qwen2-vl" in args.model_name.lower():
        replace_qwen2vl(args, model, args.method.lower())
    elif "internvl2_5" in args.model_name.lower():
        replace_internvl2_5(args, model, args.method.lower())
    else:
        raise ValueError(f"Model name {args.model_name} not supported")

# Caching model loading
# We include method and budgets in the cache key implicitly by passing args, 
# but since we modify the model in place, we should be careful.
# To ensure correctness when switching methods, we should reload/re-patch.
# Since we can't easily un-patch, we will cache the *base* model loading if possible, 
# but replace_layers modifies it. 
# So we will cache the result of loading AND patching. 
# This means if method changes, we reload. This is safer.

@st.cache_resource(show_spinner=True, max_entries=1)
def load_and_patch_model(pretrained, model_name, method, budgets, device_map, torch_dtype, max_pixels, min_pixels):
    # Create a temporary args object for replace_layers
    args = Args(
        pretrained=pretrained,
        model_name=model_name,
        method=method,
        budgets=budgets,
        device_map=device_map,
        torch_dtype=torch_dtype,
        max_pixels=max_pixels,
        min_pixels=min_pixels
    )
    
    if "llava-onevision" in model_name.lower():
        llava_model_args = {
            "attn_implementation": "eager", 
            "device_map": device_map, 
            "torch_dtype": torch_dtype,
            "multimodal": True
        }
        tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, **llava_model_args)
        model.eval()
        replace_layers(args, model)
        return model, tokenizer, image_processor, max_length, None # Extra return to match signature
        
    elif "qwen2-vl" in model_name.lower():
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            pretrained,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation="flash_attention_2", 
            local_files_only=True
        ).eval()

        qwen2vl_processor = Qwen2VLProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels, local_files_only=True)
        qwen2vl_tokenizer = AutoTokenizer.from_pretrained(pretrained, local_files_only=True)
        
        replace_layers(args, model)
        return model, qwen2vl_tokenizer, qwen2vl_processor, None, None

    elif "internvl2_5" in model_name.lower():
        model = AutoModel.from_pretrained(pretrained, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, device_map=device_map, use_flash_attn=False).eval()
        intern2_5_tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True, device_map=device_map)
        replace_layers(args, model)
        return model, intern2_5_tokenizer, None, None, None
    
    else:
        raise ValueError(f"Model name {model_name} not supported")

def run_inference_ov(args, model, tokenizer, image_processor, image):
    device = args.device
    
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
    return output[0]

def run_inference_qwen2vl(args, model, tokenizer, processor, image):
    messages = []
    context = args.question
    message = [{"role": "system", "content": "You are a helpful assistant."}]
    
    # process image
    visual = image
    base64_image = visual.convert("RGB")
    buffer = BytesIO()
    base64_image.save(buffer, format="JPEG")
    base64_bytes = base64.b64encode(buffer.getvalue())
    base64_string = base64_bytes.decode("utf-8")

    message.append({"role": "user", "content": [{"type": "image", "image": f"data:image/jpeg;base64,{base64_string}", "max_pixels": args.max_pixels, "min_pixels": args.min_pixels}, {"type": "text", "text": context}]})
    messages.append(message)
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

    if args.device_map == "auto":
        inputs = inputs.to("cuda")
    else:
        inputs = inputs.to(args.device)
    pad_token_id = tokenizer.pad_token_id

    # generate
    cont = model.generate(
        **inputs,
        eos_token_id= tokenizer.eos_token_id,
        pad_token_id=pad_token_id,
        do_sample=True if args.temperature > 0 else False,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        use_cache=args.use_cache,
    )

    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
    answers = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return answers[0]

def run_inference_internvl2_5(args, model, tokenizer, image):
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
    response, history = model.chat(tokenizer, pixel_values, contexts, gen_kwargs, num_patches_list=num_patches_list, history=None, return_history=True)
    return response

# Streamlit App
st.set_page_config(page_title="EffiVLM-Bench Inference", layout="wide")

st.title("Inference Dashboard")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Model Selection
model_options = {
    "Qwen2-VL-7B-Instruct": "/home/lyli/models/Qwen2-VL-7B-Instruct",
    "LLaVA-OneVision-7B": "/home/lyli/models/LLaVA-OneVision-7B"
}
selected_model_label = st.sidebar.selectbox("Select Pretrained Model", list(model_options.keys()))
pretrained_path = model_options[selected_model_label]

# Map to internal model_name
if "Qwen2-VL" in selected_model_label:
    model_name = "qwen2-vl"
elif "LLaVA-OneVision" in selected_model_label:
    model_name = "llava-onevision-qwen2"
else:
    model_name = "internvl2_5" # Fallback or add more options

# Method Selection
method_options = ['random', 'streamingllm', 'h2o', 'snapkv', 'look-m', 'vl-cache', 'pyramidkv', 'fastv', 'visionzip', 'prumerge+']
selected_methods = st.sidebar.multiselect("Select Methods", method_options, default=['random'])

# GPU Selection
gpu_count = st.sidebar.number_input("Number of GPUs", min_value=1, max_value=10, value=1)

# Budgets
budgets = st.sidebar.number_input("Budgets", min_value=0.0, max_value=1.0, value=0.4, step=0.05)

# Other parameters (hidden or default)
# We keep them as defaults from predict.py
args_defaults = Args()

# Main Area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input")
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    question = st.text_input("Question", value="What is shown in this image?")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=300)

with col2:
    st.subheader("Output")
    run_button = st.button("Run Inference", type="primary")
    
    output_container = st.container()

# History
if "history" not in st.session_state:
    st.session_state.history = []

if run_button and uploaded_file is not None and selected_methods:
    
    # Determine device map based on selection
    if gpu_count == 1:
        device_map = "cuda:0"
        device = "cuda:0"
    else:
        device_map = "auto"
        device = "cuda"

    for method in selected_methods:
        # Clear previous model from cache and force garbage collection to avoid OOM
        load_and_patch_model.clear()
        
        # Explicitly delete variables that might hold references to the model
        # Use try-except to handle cases where variables are not yet defined
        try: del model_components
        except NameError: pass
        try: del model
        except NameError: pass
        try: del tokenizer
        except NameError: pass
        try: del processor
        except NameError: pass
        try: del image_processor
        except NameError: pass
        
        gc.collect()
        torch.cuda.empty_cache()

        with st.spinner(f"Running inference with method: {method}..."):
            try:
                # Load model
                model_components = load_and_patch_model(
                    pretrained=pretrained_path,
                    model_name=model_name,
                    method=method,
                    budgets=budgets,
                    device_map=device_map,
                    torch_dtype=args_defaults.torch_dtype,
                    max_pixels=args_defaults.max_pixels,
                    min_pixels=args_defaults.min_pixels
                )
                
                # Prepare args
                current_args = Args(
                    pretrained=pretrained_path,
                    model_name=model_name,
                    method=method,
                    budgets=budgets,
                    question=question,
                    image_path=None, # We pass image object directly
                    device=device,
                    device_map=device_map
                )
                
                start_time = time.time()
                
                # Run inference
                if "llava-onevision" in model_name.lower():
                    model, tokenizer, image_processor, _, _ = model_components
                    output_text = run_inference_ov(current_args, model, tokenizer, image_processor, image)
                elif "qwen2-vl" in model_name.lower():
                    model, tokenizer, processor, _, _ = model_components
                    output_text = run_inference_qwen2vl(current_args, model, tokenizer, processor, image)
                elif "internvl2_5" in model_name.lower():
                    model, tokenizer, _, _, _ = model_components
                    output_text = run_inference_internvl2_5(current_args, model, tokenizer, image)
                else:
                    output_text = "Model not supported."
                    
                end_time = time.time()
                inference_time = end_time - start_time
                
                with output_container:
                    st.markdown(f"### Method: {method}")
                    st.markdown(f"**Response:**\n\n{output_text}")
                    st.info(f"Inference Time: {inference_time:.4f} seconds")
                    st.markdown("---")
                
                # Save to history
                record = {
                    "Model": selected_model_label,
                    "Method": method,
                    "Budgets": budgets,
                    "Question": question,
                    "Output": output_text,
                    "Inference Time (s)": inference_time,
                    "Image Name": uploaded_file.name,
                    "GPU Count": gpu_count
                }
                st.session_state.history.append(record)
                
            except Exception as e:
                st.error(f"An error occurred with method {method}: {e}")
                logger.exception(f"Inference failed for {method}")

# Display History
st.markdown("---")
st.subheader("Inference History")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download History as CSV",
        data=csv,
        file_name='inference_history.csv',
        mime='text/csv',
    )
else:
    st.info("No history yet. Run inference to see results here.")
