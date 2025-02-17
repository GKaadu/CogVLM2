#!/bin/sh

git pull
pkill python3
nohup torchrun --standalone --nnodes=1 --nproc-per-node=4 cli_autonomous.py --from_pretrained THUDM/cogvlm2-llama3-chat-19B --version chat_old --fp16 &
nohup python3 ./message_publisher.py &
tail -f ./nohup.out