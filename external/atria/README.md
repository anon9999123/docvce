## Environment Setup

### Create a virtual environment
```
pip install --user virtualenv virtualenvwrapper
echo export WORKON_HOME=$HOME/.virtualenvs >> ~/.bashrc
echo source ~/.local/bin/virtualenvwrapper.sh >> ~/.bashrc
source ~/.bashrc
```

```
mkvirtualenv atria
workon atria
```

### Install from source
Install the dependencies:
```
pip install -r requirements.txt
```

Build atria hydra configurations:
```
python -m atria._hydra.build_configurations
```

Setup environment variables:
```
export PYTHONPATH=<path/to/atria>/src
```
