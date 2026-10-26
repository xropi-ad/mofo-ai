This repo is to explore hands-on AI/ML techniques/practices/tooling.

Always give concise answers and if possible, don't offer too many options at once.
Try to proceed topic-by-topic, solution-by-solution.
For multi option use very short list to inform what's coming then let's discuss each one after the other.

Assume wsl2 and docker setup.
All setup should have install script. If script is complicated (loops/complex, logic) it has to be py;
othersiwe .sh is ok for trivial, sequential scripts.

For python3: use uv.
There's a setup-host folder in repo where other setup scripts live like GPU wsl2 forwarding setup, or setting up docker itself on host etc.
There's a docker folder in repo where dockerfile, docker launchers scripts live.

Py script naming: modules don't contain '-', they use '_' instead. Executable py can contain '-'.
When instructed to create script and it's for execution, claude should set exec permission on those files.
Use shebang in exec scripts.

