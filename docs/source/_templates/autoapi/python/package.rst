{% extends "python/module.rst" %}
{# Top-level packages list only their modules; subpackages sit next to them in index.rst. #}
{% block submodules %}
   {% if obj.is_top_level_object %}
      {% set visible_submodules = obj.submodules|selectattr("display")|list|sort %}
   {% else %}
      {% set visible_subpackages = obj.subpackages|selectattr("display")|list %}
      {% set visible_submodules = obj.submodules|selectattr("display")|list %}
      {% set visible_submodules = (visible_subpackages + visible_submodules)|sort %}
   {% endif %}
   {% if visible_submodules %}
Submodules
----------

.. toctree::
   :maxdepth: 1

      {% for submodule in visible_submodules %}
   {{ submodule.include_path }}
      {% endfor %}


   {% endif %}
{% endblock %}
