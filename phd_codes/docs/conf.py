# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Hyperscanning Experiments"
copyright = "2022, Ihshan Gumilar"
author = "Ihshan Gumilar"
release = "1.0.0 Exp2-redesign"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../EEG/"))
sys.path.insert(0, os.path.abspath("../eye_tracker/"))
sys.path.insert(0, os.path.abspath("../questionnaire/"))

# os.path.abspath("../EEG/")
# os.path.abspath("../eye_tracker/")
# os.path.abspath("../questionnaire/")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinxcontrib.inlinesyntaxhighlight",
]

# use language set by highlight directive if no language is set by role
inline_highlight_respect_highlight = True

# use language set by highlight directive if no role is set
inline_highlight_literals = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
