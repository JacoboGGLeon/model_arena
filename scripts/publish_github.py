"""Publish the prepared clean repository after gh authentication is available."""
from pathlib import Path
import subprocess

OWNER, REPOSITORY = 'JacoboGGLeon', 'model_arena'
ROOT = Path(__file__).resolve().parents[1]

def run(args, **kwargs):
    return subprocess.run(args, cwd=ROOT, check=True, text=True, **kwargs)

if __name__ == '__main__':
    account = run(['gh', 'api', 'user', '--jq', '.login'], capture_output=True).stdout.strip()
    if account.casefold() != OWNER.casefold():
        raise SystemExit(f'Authenticate gh as {OWNER}; current account is {account}')
    if run(['git', 'status', '--porcelain'], capture_output=True).stdout.strip():
        raise SystemExit('Commit and validate all changes before publishing')
    remote = subprocess.run(['gh', 'repo', 'view', OWNER+'/'+REPOSITORY, '--json', 'nameWithOwner'],
                            cwd=ROOT, text=True, capture_output=True)
    if remote.returncode:
        run(['gh', 'repo', 'create', OWNER+'/'+REPOSITORY, '--public',
             '--description', 'Explicit forecasting competition and observable learning histories'])
    run(['git', 'push', '-u', 'origin', 'main'])
    print('Source push completed. Verify CI before creating the PyPI release.')
