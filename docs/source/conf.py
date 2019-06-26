# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import param

sys.path.insert(0, os.path.abspath('../..'))


autodoc_mock_imports = [
    'numpy',
    'torch',
]


# -- Project information -----------------------------------------------------

project = 'pydrobert-pytorch'
copyright = '2019, Sean Robertson'
author = 'Sean Robertson'

# The full version, including alpha/beta/rc tags
release = '0.0.2'

language = 'en'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

naploeon_numpy_docstring = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    html_theme = 'default'
else:
    html_theme = 'nature'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

highlight_language = 'none'

master_doc = 'index'

# this is a hack until param issue #197 is resolved
# param.parameterized.docstring_signature = False
ipython_colours = {
    (param.ipython.red % ' ').split()[0],
    (param.ipython.green % ' ').split()[0],
    (param.ipython.blue % ' ').split()[0],
    (param.ipython.cyan % ' ').split()[0],
    (param.ipython.cyan % ' ').split()[1],
}


def my_handler(app, what, name, obj, options, lines):
    if 'Params' in name.split('.')[-1]:
        found_bounds, found_docs, idx = False, False, 0
        while idx < len(lines):
            line = lines[idx]
            if '===' in line:
                lines.pop(idx)
                continue
            if found_docs:
                for colour in ipython_colours:
                    line = line.replace(colour, '')
                if not line.strip():
                    lines.pop(idx)
                    continue
                name = line.split(':')[0].strip()
                param = obj.params()[name]
                doc = param.doc
                deft = param.default
                bounds = param.bounds if hasattr(param, 'bounds') else None
                lines[idx] = '**{}**: {} *default={}{}*'.format(
                    name, doc, deft,
                    ', bounds={}'.format(bounds) if bounds else '')
                lines.insert(idx + 1, '')
                lines.insert(idx + 1, '')
                idx += 3
            elif 'Parameter docstrings' in line:
                found_docs = True
                lines[idx] = ''
                idx += 1
            else:
                lines.pop(idx)
        options['undoc-members'] = False


def setup(app):
    app.connect('autodoc-process-docstring', my_handler)
