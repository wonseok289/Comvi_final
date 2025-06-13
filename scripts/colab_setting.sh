#!/bin/bash

# ���̽� ��ġ
# sudo apt update
# sudo apt install -y software-properties-common
# sudo add-apt-repository ppa:deadsnakes/ppa -y
# sudo apt update
# sudo apt install -y python3.10 python3.10-venv python3.10-dev

# # ����ȯ�� ����� �� Ȱ��ȭ
# python3.10 -m venv venv
# source venv/bin/activate

conda create -n venv python=3.10
conda init bash
source ~/.bashrc
conda activate venv
rm colab_setup.ipynb

# ��� �̵�
cd Comvi_final

# pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124


# �ʿ� ��Ű�� ��ġ
pip install -r requirements.txt

python -m ipykernel install --user --name venv --display-name venv

# # wandb �α���
# pip install wandb
# wandb login

# sh scripts/download_coco.sh
# sh scripts/download_coco_test.sh
# sh scripts/create_coco128_split.sh
# sh scripts/download_carvana.sh
# sh scripts/download_cityscapes.sh

