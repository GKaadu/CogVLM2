#!/bin/sh
cd /home/ubuntu/CogVLM2
git pull
pkill python3
sudo systemctl restart aimessage.service
/home/ubuntu/.venv/bin/torchrun --standalone --nnodes=1 --nproc-per-node=1 /home/ubuntu/CogVLM2/cli_autonomous.py --model THUDM/cogvlm2-llama3-chat-19B