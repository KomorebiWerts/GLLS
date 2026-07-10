"""Compatibility shims for the repository's Gradio 4.44 frontends."""

from __future__ import annotations


def _boolean_schema_python_type(schema: bool) -> str:
    return "Any" if schema else "Never"


def _patch_starlette_template_response() -> None:
    """Support Gradio's legacy TemplateResponse call on newer Starlette."""
    try:
        import inspect

        from starlette.templating import Jinja2Templates
    except Exception:
        return

    current = getattr(Jinja2Templates, "TemplateResponse", None)
    if getattr(current, "_glls_compat_patched", False):
        return
    try:
        params = list(inspect.signature(current).parameters)
    except (TypeError, ValueError):
        return
    if len(params) < 3 or params[1] != "request":
        return

    original = current

    def compat_template_response(self, *args, **kwargs):
        if args and isinstance(args[0], str):
            name = args[0]
            context = args[1] if len(args) > 1 else kwargs.pop("context", None)
            remaining = args[2:]
            if context is None:
                context = {}
            request = kwargs.pop("request", None)
            if request is None and isinstance(context, dict):
                request = context.get("request")
            return original(self, request, name, context, *remaining, **kwargs)
        return original(self, *args, **kwargs)

    compat_template_response._glls_compat_patched = True
    Jinja2Templates.TemplateResponse = compat_template_response


def _patch_gradio_client_boolean_schema() -> None:
    """Handle boolean JSON schemas emitted by current Pydantic releases."""
    try:
        from gradio_client import utils as client_utils
    except Exception:
        return

    private = getattr(client_utils, "_json_schema_to_python_type", None)
    if private is not None and not getattr(private, "_glls_compat_patched", False):
        def compat_private_schema_type(schema, defs):
            if isinstance(schema, bool):
                return _boolean_schema_python_type(schema)
            return private(schema, defs)

        compat_private_schema_type._glls_compat_patched = True
        client_utils._json_schema_to_python_type = compat_private_schema_type

    public = getattr(client_utils, "json_schema_to_python_type", None)
    if public is not None and not getattr(public, "_glls_compat_patched", False):
        def compat_public_schema_type(schema):
            if isinstance(schema, bool):
                return _boolean_schema_python_type(schema)
            return public(schema)

        compat_public_schema_type._glls_compat_patched = True
        client_utils.json_schema_to_python_type = compat_public_schema_type


def patch_gradio_compat() -> None:
    """Apply the compatibility fixes needed before launching a Gradio app."""
    _patch_starlette_template_response()
    _patch_gradio_client_boolean_schema()
