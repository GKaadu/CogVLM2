#!/bin/sh

git pull
pkill python3
nohup torchrun --standalone --nnodes=1 --nproc-per-node=1 cli_autonomous.py --model THUDM/cogvlm2-llama3-chat-19B &
nohup python3 ./message_publisher.py &
tail -f ./nohup.out