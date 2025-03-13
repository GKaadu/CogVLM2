#!/bin/sh
sudo systemctl restart rabbitmq-server
/home/ubuntu/.venv/bin/python3 /home/ubuntu/CogVLM2/message_publisher.py