# MHC Dataset

## Setup

Create a virtual environment and install the dependencies.

```bash
python -m venv venv
source venv/bin/activate
```

Install the dependencies.
```bash
pip install -r requirements.txt
```
Install the package in editable mode to allow for development and testing without import issues.
```bash
pip install -e .
```

Install the SlurmMultiNodePool package to make distributing the code on a cluster simple.
```bash
pip install git+https://github.com/NarayanSchuetz/SlurmMultiNodePool.git
```

