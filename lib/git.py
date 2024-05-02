import git
from pathlib import Path

# git_root = Path(git_repo.git.rev_parse("--show-toplevel"))


def git_repo():
    try:
        return git.Repo(Path(__file__).parent, search_parent_directories=True)
    except git.exc.InvalidGitRepositoryError:
        return None


def is_git_repo():
    try:
        git.Repo(Path(__file__).parent, search_parent_directories=True)
    except git.exc.InvalidGitRepositoryError:
        return False
    return True


def git_repo():
    try:
        repo = git.Repo(Path(__file__).parent, search_parent_directories=True)
        return repo
    except git.exc.InvalidGitRepositoryError:
        return None


def get_rev_short():
    if not is_git_repo():
        return "no_git_repo"
    git_repo = git.Repo(Path(__file__).parent, search_parent_directories=True)
    sha = git_repo.head.commit.hexsha
    return git_repo.git.rev_parse(sha, short=7)


def get_rev():
    if not is_git_repo():
        return "no_git_repo"
    git_repo = git.Repo(Path(__file__).parent, search_parent_directories=True)
    sha = git_repo.head.commit.hexsha
    return git_repo.git.rev_parse(sha)
