from time import sleep
from rich.progress import Progress

total_steps = 100

def run_progress_bar():
    with Progress() as progress:
        task = progress.add_task("[cyan]Processing...", total=total_steps)

        for step in range(total_steps):
            sleep(0.01)  # Simulate work being done
            progress.update(task, advance=1)


