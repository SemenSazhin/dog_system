#!/usr/bin/env python3
"""Разовый запуск на своём компьютере: получает GOOGLE_REFRESH_TOKEN.

pip install google-auth-oauthlib
python scripts/get_refresh_token.py client_secret.json
"""
import sys

from google_auth_oauthlib.flow import InstalledAppFlow

flow = InstalledAppFlow.from_client_secrets_file(
    sys.argv[1], scopes=["https://www.googleapis.com/auth/drive.file"])
creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")
print("\nGOOGLE_CLIENT_ID     =", creds.client_id)
print("GOOGLE_CLIENT_SECRET =", creds.client_secret)
print("GOOGLE_REFRESH_TOKEN =", creds.refresh_token)
