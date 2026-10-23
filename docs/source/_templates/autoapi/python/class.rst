{% if obj.display %}
   {% if is_own_page %}
{{ obj.id }}
{{ "=" * obj.id | length }}

   {% endif %}
   {% set visible_children = obj.children|selectattr("display")|list %}
   {% set own_page_children = visible_children|selectattr("type", "in", own_page_types)|list %}
   {% if is_own_page and own_page_children %}
.. toctree::
   :hidden:

      {% for child in own_page_children %}
   {{ child.include_path }}
      {% endfor %}

   {% endif %}
.. py:{{ obj.type }}:: {% if is_own_page %}{{ obj.id }}{% else %}{{ obj.short_name }}{% endif %}{% if obj.type_params %}[{{ obj.type_params }}]{% endif %}{% if obj.args %}({{ obj.args }}){% endif %}

   {% for (args, return_annotation) in obj.overloads %}
      {{ " " * (obj.type | length) }}   {{ obj.short_name }}{% if args %}({{ args }}){% endif %}

   {% endfor %}
   {% if obj.bases %}
      {% if "show-inheritance" in autoapi_options %}

   Bases: {% for base in obj.bases %}{{ base|link_objs }}{% if not loop.last %}, {% endif %}{% endfor %}
      {% endif %}


      {% if "show-inheritance-diagram" in autoapi_options and obj.bases != ["object"] %}
   .. autoapi-inheritance-diagram:: {{ obj.obj["full_name"] }}
      :parts: 1
         {% if "private-members" in autoapi_options %}
      :private-bases:
         {% endif %}

      {% endif %}
   {% endif %}
   {% if obj.docstring %}

   {{ obj.docstring|indent(3) }}
   {% endif %}
   {# nest.grad: gradients named _<method>_grad_<variable> are listed under <method>. #}
   {% set method_names = visible_children|selectattr("type", "equalto", "method")|map(attribute="short_name")|list %}
   {% for obj_item in visible_children %}
      {% set grad_parts = obj_item.short_name[1:].split("_grad_") %}
      {% set is_gradient = obj_item.type == "method" and obj_item.short_name.startswith("_") and grad_parts|length == 2 and grad_parts[0] in method_names %}
      {% if obj_item.type not in own_page_types and not is_gradient %}

   {{ obj_item.render()|indent(3) }}
         {% if obj_item.type == "method" %}
            {% set grad_prefix = "_" ~ obj_item.short_name ~ "_grad_" %}
            {% for gradient in visible_children|selectattr("type", "equalto", "method") %}
               {% if gradient.short_name.startswith(grad_prefix) %}

      .. py:method:: {{ obj_item.short_name }}.grad_{{ gradient.short_name[grad_prefix|length:] }}({{ gradient.args }})

         {{ gradient.docstring|indent(9) }}
               {% endif %}
            {% endfor %}
         {% endif %}
      {% endif %}
   {% endfor %}
   {% if is_own_page and own_page_children %}
      {% set visible_attributes = own_page_children|selectattr("type", "equalto", "attribute")|list %}
      {% if visible_attributes %}
Attributes
----------

.. autoapisummary::

         {% for attribute in visible_attributes %}
   {{ attribute.id }}
         {% endfor %}


      {% endif %}
      {% set visible_exceptions = own_page_children|selectattr("type", "equalto", "exception")|list %}
      {% if visible_exceptions %}
Exceptions
----------

.. autoapisummary::

         {% for exception in visible_exceptions %}
   {{ exception.id }}
         {% endfor %}


      {% endif %}
      {% set visible_classes = own_page_children|selectattr("type", "equalto", "class")|list %}
      {% if visible_classes %}
Classes
-------

.. autoapisummary::

         {% for klass in visible_classes %}
   {{ klass.id }}
         {% endfor %}


      {% endif %}
      {% set visible_methods = own_page_children|selectattr("type", "equalto", "method")|list %}
      {% if visible_methods %}
Methods
-------

.. autoapisummary::

            {% for method in visible_methods %}
   {{ method.id }}
            {% endfor %}


      {% endif %}
   {% endif %}
{% endif %}
