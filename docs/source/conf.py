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

param.parameterized.docstring_signature = False
param.parameterized.docstring_describe_params = False

sys.path.insert(0, os.path.abspath("../../src"))


# -- Project information -----------------------------------------------------

project = "pydrobert-pytorch"
copyright = "2023, Sean Robertson"
author = "Sean Robertson"

language = "en"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = True
autodoc_mock_imports = [
    "numpy",
    "torch",
]
autodoc_typehints = "none"
autodoc_type_aliases = napoleon_type_aliases = {"np.ndarray": "numpy.ndarray"}
autodoc_inherit_docstrings = False
napoleon_preprocess_types = True
typehints_document_rtype = False
napoleon_use_rtype = False
napoleon_custom_sections = [
    ("Call Parameters", "returns_style"),
    ("Variables", "returns_style"),
]

intersphinx_mapping = {
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "pydrobert.kaldi": ("https://pydrobert-kaldi.readthedocs.io/en/latest", None),
    "pydrobert.param": ("https://pydrobert-param.readthedocs.io/en/latest", None),
    "pydrobert.speech": ("https://pydrobert-speech.readthedocs.io/en/latest", None),
    "python": ("https://docs.python.org/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# -- Options for HTML output -------------------------------------------------

# on_rtd = os.environ.get("READTHEDOCS") == "True"
# if on_rtd:
#     html_theme = "default"
# else:
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

highlight_language = "none"

master_doc = "index"

# this is a hack until param issue #197 is resolved
ipython_colours = {
    (param.ipython.red % " ").split()[0],
    (param.ipython.green % " ").split()[0],
    (param.ipython.blue % " ").split()[0],
    (param.ipython.cyan % " ").split()[0],
    (param.ipython.cyan % " ").split()[1],
}


def docstring_handler(app, what, name, obj, options, lines):
    if "Params" in name.split(".")[-1]:
        pdict = obj.param.objects(instance=False)
        del pdict["name"]
        new_lines = []
        for name, p in pdict.items():
            doc = p.doc
            deft = p.default
            bounds = p.bounds if hasattr(p, "bounds") else None
            new_lines.append(
                "- **{}**: {}. *default={}{}*".format(
                    name, doc, deft, ", bounds={}".format(bounds) if bounds else ""
                )
            )
            new_lines.append("")
            new_lines.append("")
        if new_lines:
            new_lines.insert(0, "")
            new_lines.insert(0, "")
            new_lines.insert(1, "**Parameters**")
            new_lines.insert(2, "")
            new_lines.insert(2, "")
            lines += new_lines
        options["undoc-members"] = False


# def preprocess_signature(app, obj, bound_method):
#     import inspect

#     print(obj, inspect.signature(obj))


def setup(app):
    # app.connect("autodoc-before-process-signature", preprocess_signature)
    app.connect("autodoc-process-docstring", docstring_handler)
