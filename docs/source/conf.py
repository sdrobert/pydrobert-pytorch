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
import future.utils

param.parameterized.docstring_signature = False
param.parameterized.docstring_describe_params = False

# this saves us when dealing with with_metaclass on mocked objects

def with_metaclass(meta, *bases):
    class metaclass(meta):
        __call__ = type.__call__
        __init__ = type.__init__
        def __new__(cls, name, this_bases, d):
            if this_bases is None:
                return type.__new__(cls, name, (), d)
            return meta(name, bases, d)
    try:
        v = metaclass('temporary_class', None, {})
        class Dummy(v):
            pass
    except TypeError:
        # probably a mock object. In this case, return the first base
        v = bases[0] if len(bases) else object
    return v

future.utils.with_metaclass = with_metaclass

sys.path.insert(0, os.path.abspath('../..'))


autodoc_mock_imports = [
    'numpy',
    'torch',
]

# -- Project information -----------------------------------------------------

project = 'pydrobert-pytorch'
copyright = '2019, Sean Robertson'
author = 'Sean Robertson'

language = 'en'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
]

naploeon_numpy_docstring = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

intersphinx_mapping = {
    'torch': ('https://pytorch.org/docs/stable/', None),
    'python': ('https://docs.python.org/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'pydrobert.param': (
        'https://pydrobert-param.readthedocs.io/en/stable/', None),
}


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

highlight_language = 'default'

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
        pdict = obj.param.objects(instance=False)
        del pdict['name']
        new_lines = []
        for name, p in pdict.items():
            doc = p.doc
            deft = p.default
            bounds = p.bounds if hasattr(p, 'bounds') else None
            new_lines.append('- **{}**: {}. *default={}{}*'.format(
                name, doc, deft,
                ', bounds={}'.format(bounds) if bounds else ''))
            new_lines.append('')
            new_lines.append('')
        if new_lines:
            new_lines.insert(0, '')
            new_lines.insert(0, '')
            new_lines.insert(1, '**Parameters**')
            new_lines.insert(2, '')
            new_lines.insert(2, '')
            lines += new_lines
        options['undoc-members'] = False


def setup(app):
    app.connect('autodoc-process-docstring', my_handler)
