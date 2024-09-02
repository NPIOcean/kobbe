# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

project = 'kobbe'
copyright = '2024, Øyvind Foss'
author = 'Øyvind Foss'
release = '0.0.1'



sys.path.insert(0, os.path.abspath('../src/kobbe/'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.coverage',
              'sphinx.ext.napoleon',
              'myst_parser',
              'sphinx.ext.mathjax',]  # For LaTeX support]


myst_enable_extensions = [
    "dollarmath",   # Enables LaTeX math using $...$
]

templates_path = ['_templates']
exclude_patterns = []


# Set the logo
html_logo = "_static/logos/kobbe_logo.png"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
