#!/bin/sh
pwd
cd /home/ubuntu/CogVLM2
git pull
pkill python3
nohup /home/ubuntu/.venv/bin/torchrun --standalone --nnodes=1 --nproc-per-node=1 /home/ubuntu/CogVLM2/cli_autonomous.py --model THUDM/cogvlm2-llama3-chat-19B &
nohup /home/ubuntu/.venv/bin/python3 /home/ubuntu/CogVLM2/message_publisher.py &
tail -f /home/ubuntu/CogVLM2/nohup.out