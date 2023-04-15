import git
from pathlib import Path

GIT_REPO = git.Repo(Path(__file__).parent, search_parent_directories=True)
GIT_ROOT = Path(GIT_REPO.git.rev_parse("--show-toplevel"))
