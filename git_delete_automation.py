import requests 
import git_info 

GITHUB_USERNAME: str = git_info.Git_username
GITHUB_TOKEN: str = git_info.Git_Tokens

# make file git_info add your tokens and username in the format
# Git_username = "your_github_username"
# Git_Tokens = "ghp_your_github_token"

REPOS_TO_DELETE = [
]

headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

for repo in REPOS_TO_DELETE:
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo}"
    response = requests.delete(url, headers=headers)
    if response.status_code == 204:
        print(f"✅ Deleted: {repo}")
    elif response.status_code == 404:
        print(f"⚠️ Not found: {repo}")
    else:
        print(f"❌ Failed to delete {repo} — Status code: {response.status_code}")
        print(response.text)
