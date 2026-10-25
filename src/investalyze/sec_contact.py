"""The contact address that SEC asks for in the User-Agent of every request: SEC_CONTACT_EMAIL in .env in the app folder
(not in git, the repo is public). SEC refuses a request without a real contact address (HTTP 403)."""

import os

from dotenv import dotenv_values

from investalyze import paths


def user_agent(purpose: str) -> str:
    """The User-Agent of a SEC request, for example 'investalyze ingest (name@example.com)'."""
    email = os.environ.get('SEC_CONTACT_EMAIL') or dotenv_values(paths.APP_DIR / '.env').get('SEC_CONTACT_EMAIL')
    if not email:
        raise RuntimeError(f'SEC_CONTACT_EMAIL is not set: put your e-mail address in {paths.APP_DIR / ".env"} '
                           '(SEC_CONTACT_EMAIL=name@example.com); SEC refuses requests without a contact address.')
    return f'investalyze {purpose} ({email})'
