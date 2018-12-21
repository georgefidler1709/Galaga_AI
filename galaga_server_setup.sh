scp -i "~/.ssh/gej_personal.pem" galaga_AI.py ubuntu@ec2-54-145-145-123.compute-1.amazonaws.com:./

scp -i "~/.ssh/gej_personal.pem" Galaga.nes ubuntu@ec2-54-145-145-123.compute-1.amazonaws.com:./

sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt install virtualenv
virtualenv --system-site-packages -p python3 ./venv
source ./venv/bin/activate
pip install --upgrade pip
pip install --upgrade tensorflow
pip install gym-retro
mkdir galaga
mv galaga_AI.py galaga
mv Galaga.nes galaga
cd galaga
python3 -m retro.import .

