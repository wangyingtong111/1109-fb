"""
An extension to the Django web framework that provides version control for model instances.

Developed by Dave Hall.

<http://www.etianen.com/>
"""

from reversion.errors import (  # noqa
    RevertError,
    RevisionManagementError,
    RegistrationError,
)
from reversion.revisions import (  # noqa
    is_active,
    is_manage_manually,
    get_user,
    set_user,
    get_comment,
    set_comment,
    get_date_created,
    set_date_created,
    add_meta,
    add_to_revision,
    create_revision,
    register,
    is_registered,
    unregister,
    get_registered_models,
)

__version__ = VERSION = (6, 5, 2)


def __getattr__(name):
    # Lazily expose revert_group without importing reversion.models (which
    # requires the app registry to be ready) at package import time.
    if name == "revert_group":
        from reversion.models import revert_group
        return revert_group
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
