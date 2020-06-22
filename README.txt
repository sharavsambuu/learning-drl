# Dependencies
  sudo apt install libsdl2-dev swig python3-tk

# Prepare
  virtualenv -p python3 env && source env/bin/activate && pip install -r requirements.txt

# Additional Rocket Lander Gym extension
  git clone https://github.com/Jeetu95/Rocket_Lander_Gym.git
  cd Rocket_Lander_Gym && pip install .
