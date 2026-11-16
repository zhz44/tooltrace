"""Python imports a module named ``sitecustomize`` at start-up if it finds one.

``tooltrace run`` puts this directory at the front of PYTHONPATH, so the agent's
Python process imports this file and the hooks are installed before the agent
starts. Without TOOLTRACE_PATH in the environment nothing happens.
"""

from tooltrace.hook import install_from_env

install_from_env()
