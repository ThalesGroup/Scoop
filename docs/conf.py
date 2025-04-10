# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Scoop'
copyright = '2025, Nathan Rousselot, Karine Heydemann, Loïc Masure and Vincent Migairou'
author = 'Nathan Rousselot, Karine Heydemann, Loïc Masure and Vincent Migairou'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Adjust the relative path to point to your project root.

extensions = [
    'sphinx.ext.autodoc',      # Automatically document your modules
]

import os
import sys
sys.path.insert(0, os.path.abspath('../'))  # Adjust the relative path to point to your project root.

extensions = [
    'sphinx.ext.autodoc',      # Automatically document your modules
]
