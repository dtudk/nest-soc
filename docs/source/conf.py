# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "nest"
copyright = "2026, Danmark Tekniske Universitet"
author = "Rafael Nogueira Nakashima, Javid Beyrami"
release = "0.3.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"

# -- Auto-api setup ----------------------------------------------------------

autoapi_dirs = ["../../nest"]
autoapi_template_dir = "_templates/autoapi"  # index.rst lists subpackages next to nest
autoapi_ignore = ["*/examples/*"]
autoapi_options = [
    "members",
    "undoc-members",
    "private-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
]  # default minus 'imported-members'


def _skip_private(app, what, name, obj, skip, options):
    # "private-members" is needed for the nested _<method>_grad_<v> gradients,
    # but other private members and modules (e.g. nest.grad._utils) are hidden.
    short_name = name.rsplit(".", 1)[-1]
    private = short_name.startswith("_") and not short_name.startswith("__")
    if private and (what != "method" or "_grad_" not in short_name):
        return True
    return skip


def setup(app):
    app.connect("autoapi-skip-member", _skip_private)
