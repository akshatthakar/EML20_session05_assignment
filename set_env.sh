#!/bin/bash

aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID" --profile emlov2 && aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY" --profile emlov2 && aws configure set region "$AWS_REGION" --profile emlov2 && aws configure set output "text" --profile emlov2

mkdir logs

aws s3 sync s3://emlov2session5/logs logs

python demo.py ckpt_path=logs/train/runs/2022-10-02_16-56-52/model.script.pt