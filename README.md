# wireless-FL

## Setting up development environment
[![Python 3.8, 3.10](https://img.shields.io/badge/python-3.8,3.10-blue.svg)](https://www.python.org/downloads/release/python-360/)
## 1. Setting up development environment
let's have fun and then complete this part

### Python Version Preparation
### Package manager

#### Poetry (is preferred)

To manage virtual environment in this project, you should install [poetry](https://python-poetry.org/)
with the following command:

```
pip install poetry==1.5.0
```

**Note: If you have Keyring, please disable it by this command before installing dependencies:**

```
keyring --disable
```

To activate or create the virtualenv use the following command:

```
poetry env use <python-version>
poetry shell
```

and, for installation of the dependencies, use this command at the root of the project:

```
poetry install
```

To install a new package, use this command:

```
poetry add <pip_package_name>==<package-version>
```

#### venv

create a virtual environment:

```
virtualenv venv
```

activate it:

```
source venv/bin/activate
```

install requirements:

```
pip install -r requirements.txt
```

To install a new package, use this command:

```
pip install <pip_package_name>==<package-version>
```

