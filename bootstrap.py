"""
AEGIS Bootstrap
Version: 0.1.0

Artificial Intelligence Framework for
Evaluating and Guarding Information Integrity and Security
"""

from pathlib import Path

VERSION = "0.1.0"
PROJECT_NAME = "AEGIS"


def banner():
    print("=" * 60)
    print("AEGIS Bootstrap")
    print(f"Version {VERSION}")
    print("Artificial Intelligence Framework for")
    print("Evaluating and Guarding Information Integrity and Security")
    print("=" * 60)


def main():
    banner()
    print("\nBootstrap initialized successfully.")


if __name__ == "__main__":
    main()