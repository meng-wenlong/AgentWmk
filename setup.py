"""
Setup script with custom post-install command for SWE-agent.
This is used alongside pyproject.toml for custom installation steps.
"""

import os
import subprocess
import sys
from pathlib import Path
from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info


class CustomInstallCommand(install):
    """Custom installation command that sets up SWE-agent."""

    def run(self):
        # Run the standard installation
        install.run(self)
        # Run post-installation setup
        self.setup_sweagent()

    def setup_sweagent(self):
        """Clone and install SWE-agent repository."""
        print("\n" + "="*50)
        print("Setting up SWE-agent...")
        print("="*50 + "\n")

        home = Path.home()
        sweagent_dir = home / "SWE-agent"

        # Check if already exists
        if not sweagent_dir.exists():
            print(f"Cloning SWE-agent repository to {sweagent_dir}...")
            try:
                subprocess.run([
                    "git", "clone",
                    "https://github.com/SWE-agent/SWE-agent.git",
                    str(sweagent_dir)
                ], check=True)
                print("✓ Repository cloned successfully")
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to clone SWE-agent: {e}")
                return
        else:
            print(f"✓ SWE-agent repository already exists at {sweagent_dir}")

        # Install SWE-agent in editable mode
        print("Installing SWE-agent in development mode...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-e", str(sweagent_dir)
            ], check=True)
            print("✓ SWE-agent installed successfully")
            print("\n" + "="*50)
            print("✅ Setup complete!")
            print("SWE-agent has been installed in development mode.")
            print("No environment variables needed - it will work directly!")
            print("="*50 + "\n")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to install SWE-agent: {e}")
            return


class CustomDevelopCommand(develop):
    """Custom develop command that sets up SWE-agent."""

    def run(self):
        # Run the standard develop installation
        develop.run(self)
        # Run post-installation setup
        CustomInstallCommand.setup_sweagent(self)


class CustomEggInfoCommand(egg_info):
    """Custom egg_info command that ensures SWE-agent is available."""

    def run(self):
        # Run the standard egg_info
        egg_info.run(self)


# Setup function that will be called by pip
# Note: When using pyproject.toml with pip install -e .,
# these custom commands may not be triggered automatically.
# You may need to run: python setup.py develop
# Or manually install SWE-agent: cd ~/SWE-agent && pip install -e .
setup(
    cmdclass={
        'install': CustomInstallCommand,
        'develop': CustomDevelopCommand,
        'egg_info': CustomEggInfoCommand,
    },
)