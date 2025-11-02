# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Powergrid'
copyright = '2025, Philemon Maangi'
author = 'Philemon Maangi'
release = 'v2.0.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


extensions = [
    'myst_parser',
    'sphinxcontrib.mermaid',
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx_autodoc_typehints',
]

myst_enable_extensions = [
    "colon_fence",  # allows ::: blocks
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': 4,
}

html_static_path = ['_static']
