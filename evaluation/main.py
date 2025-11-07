from lm_eval.tasks import TaskManager

task_manager = TaskManager(verbosity="DEBUG", include_path="./evaluation/tasks")
tasks = task_manager.list_all_tasks(False, False, True)

assert tasks.find("spider2_local") != -1, "Spider2 task not found"
assert tasks.find("bird_mini_dev") != -1, "Bird mini-dev task not found"
