# Dependencies
  sudo apt install libsdl2-dev swig python3-tk
  sudo apt install python-numpy cmake zlib1g-dev libjpeg-dev libboost-all-dev gcc libsdl2-dev wget unzip

# Prepare
  virtualenv -p python3 env && source env/bin/activate && pip install -r requirements.txt

# Additional Rocket Lander Gym extension
  git clone https://github.com/Jeetu95/Rocket_Lander_Gym.git

  change CONTINUOUS variable in Rocket_Lander_Gym/rocket_lander_gym/envs/rocket_lander.py to False

  cd Rocket_Lander_Gym && pip install .
