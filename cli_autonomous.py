# -*- encoding: utf-8 -*-
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import time
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map

import pika
import json
from PIL import Image
import requests
from io import BytesIO

def get_next_message():
    credentials = pika.PlainCredentials('guest', 'guest')
    parameters = pika.ConnectionParameters(host='localhost', credentials=credentials)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()

    channel.queue_declare(queue='chat_queue', durable=False)

    method_frame, header_frame, body = channel.basic_get(queue='chat_queue')
    connection.close()
    if method_frame:
        channel.basic_ack(method_frame.delivery_tag)
        return json.loads(body)
    else:
        return None
    

def post_reply(response, history, request_message_id):
    credentials = pika.PlainCredentials('guest', 'guest')
    parameters = pika.ConnectionParameters(host='localhost', credentials=credentials)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()

    channel.queue_declare(queue='reply_queue', durable=False)

    message = {
        'response': response,
        'history': history,
        'request_id': request_message_id
    }
    channel.basic_publish(exchange='', routing_key='reply_queue', body=json.dumps(message))

    connection.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=2048, help='max length of the total sequence')
    parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
    parser.add_argument("--top_k", type=int, default=1, help='top k for top k sampling')
    parser.add_argument("--temperature", type=float, default=.8, help='temperature for sampling')
    parser.add_argument("--chinese", action='store_true', help='Chinese interface')
    parser.add_argument('--quant', type=int, choices=[4, 8], default=0, help='Enable 4-bit or 8-bit precision loading')

    parser.add_argument("--model", type=str, default="THUDM/cogvlm2-llama3-chat-19B", help='huggingface model name')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--stream_chat", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    args = parser.parse_args()

    print("rank",rank)
    print("world_size",world_size)

    MODEL_PATH = args.model
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
        0] >= 8 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=TORCH_TYPE,
        trust_remote_code=True
    )

    num_gpus = torch.cuda.device_count()
    max_memory_per_gpu = "19GiB"

    device_map = infer_auto_device_map(
        model=model,
        max_memory={i: max_memory_per_gpu for i in range(num_gpus)},
        no_split_module_classes=["CogVLMDecoderLayer"]
    )

    model = dispatch_model(model, device_map=device_map)
    model = model.eval()

    if rank == 0:
        print('*********** LISTENING FOR REQUESTS ***********')
        
    while True:
        time.sleep(0.5)
        history = None
        
        if rank == 0:
            next_message = get_next_message()
            if next_message is None:
                continue
            image_path = next_message.get('image_path', '')
            print('Message received: ' + next_message['id'])
            is_valid = get_valid_image(image_path)
            if not is_valid:
                post_reply('Not a valid image: ' + image_path, [], next_message['id'])
                continue
        else:
            image_path = None

        if world_size > 1:
            image_path_broadcast_list = [image_path]
            # todo, what is torch.distributed.broadcast_object_list?
            torch.distributed.broadcast_object_list(image_path_broadcast_list, 0)
            image_path = image_path_broadcast_list[0]

        assert image_path is not None

        if rank == 0:
            query = next_message.get('query', '')
        else:
            query = None
            
        if world_size > 1:
            query_broadcast_list = [query]
            torch.distributed.broadcast_object_list(query_broadcast_list, 0)
            query = query_broadcast_list[0]
        
        assert query is not None
            
        if rank == 0:
            history = next_message.get('history', [])
        else:
            history = []
        
        if world_size > 1:
            history_broadcast_list = [json.dumps(history)]
            torch.distributed.broadcast_object_list(history_broadcast_list, 0)
            history = json.loads(history_broadcast_list[0])
            
        try:
            response = call_ai(image_path, model, tokenizer, query, history, DEVICE, TORCH_TYPE)
        except Exception as e:
            print(e)
            break
        if rank == 0:
            post_reply(response, history, next_message['id'])


def call_ai (image_path, model, tokenizer, query, history, DEVICE, TORCH_TYPE):
    print('Calling ai with query:' +query)
    if image_path is None:
        input_by_model = model.build_conversation_input_ids(
            tokenizer,
            query=query,
            history=history,
            template_version='chat'
        )
    else:
        image = get_image(image_path)
        if image is None:
            print('Image did not get retrived')
        print('about to build model version ids')
        input_by_model = model.build_conversation_input_ids(
            tokenizer,
            query=query,
            history=history,
            images=[image],
            template_version='chat'
        )
    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
        'images': [[input_by_model['images'][0].to(DEVICE).to(TORCH_TYPE)]] if image_path is not None else None,
    }
    gen_kwargs = {
        "max_new_tokens": 2048,
        "pad_token_id": 128002,
        "top_k": 1,
    }
    with torch.no_grad():
        print('about to generate')
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\nCogVLM2:", response)
        image.close()
        return response


def get_valid_image(image_path):
    try:
        # Check if image_path is a URL
        if image_path.startswith('http://') or image_path.startswith('https://'):
            response = requests.get(image_path)
            response.raise_for_status()  # Raise an error for bad status codes
            with Image.open(BytesIO(response.content)) as img:
                img.verify()  # Verify that it's an image
                img.close()
                return True
        else:
            # Local file path
            with Image.open(image_path) as img:
                img.verify()  # Verify that it's an image
                img.close()
                return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def get_image(image_path):
    try:
        # Check if image_path is a URL
        if image_path.startswith('http://') or image_path.startswith('https://'):
            response = requests.get(image_path)
            response.raise_for_status()  # Raise an error for bad status codes
            with Image.open(BytesIO(response.content)).convert('RGB') as img:
                return img
        else:
            # Local file path
            with Image.open(image_path).convert('RGB') as img:
                return img
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    main()
