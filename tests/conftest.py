"""Shared fixtures; forces a minimal valid environment for every test."""

import os

os.environ.setdefault("GOOGLE_API_KEY", "test-key-not-real")
os.environ.setdefault("LOG_LEVEL", "WARNING")
