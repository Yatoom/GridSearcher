import tasks

# Settings
python_path = "/home/jhoof/python/python36/bin/python3"
api_path = "/home/jhoof/GridSearcher/cli.py"
config = "#!/bin/sh\n#SBATCH -t 6:00:00 -N=1 --constraint=avx2"

tasks = tasks.tasks

# Create jobs for each task
for task_id in tasks:
    with open(f"jobs/{task_id}.sh", "w+") as f:
        description = f"{config}\n{python_path} {api_path} {task_id}"
        f.write(description)
