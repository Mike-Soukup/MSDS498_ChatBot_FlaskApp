# Echo commands and fail on error
set -ev

# Install or update needed software
sudo apt-get update
sudo apt-get install -yq git supervisor python python-pip python3-distutils
pip install --upgrade pip virtualenv

# Fetch source code
export HOME=/root
git clone git@github.com:Mike-Soukup/MSDS498_ChatBot_FlaskApp.git

# Install Cloud Ops Agent
sudo bash MSDS498_ChatBot_FlaskApp/add-google-cloud-ops-agent-repo.sh --also-install

# Account to own server process
useradd -m -d /home/pythonapp pythonapp

# Python environment setup
virtualenv -p python3 MSDS498_ChatBot_FlaskApp/env
/bin/bash -c "source MSDS498_ChatBot_FlaskApp/env/bin/activate"
MSDS498_ChatBot_FlaskApp/env/bin/pip install -r /MSDS498_ChatBot_FlaskApp/requirements.txt

# Set ownership to newly created account
chown -R pythonapp:pythonapp  

# Put supervisor configuration in proper place
cp  /MSDS498_ChatBot_FlaskApp/python-app.conf /etc/supervisor/conf.d/python-app.conf

# Start service via supervisorctl
supervisorctl reread
supervisorctl update