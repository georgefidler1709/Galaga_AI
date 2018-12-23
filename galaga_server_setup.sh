scp -i "~/.ssh/gej_personal.pem" galaga_AI.py ubuntu@ec2-34-202-237-200.compute-1.amazonaws.com:./galaga

scp -i "~/.ssh/gej_personal.pem" Galaga.nes ubuntu@ec2-34-202-237-200.compute-1.amazonaws.com:./galaga

sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt install virtualenv
virtualenv --system-site-packages -p python3 ./venv
source ./venv/bin/activate
pip install --upgrade pip
pip install --upgrade tensorflow
pip install gym-retro
mkdir galaga
cd galaga
python3 -m retro.import .

