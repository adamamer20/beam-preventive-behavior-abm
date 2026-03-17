from __future__ import annotations

from .handlers_encoding import DataCleaningEncodingImputationMixin
from .handlers_transforms import DataCleaningTransformHandlersMixin


class DataCleaningHandlersMixin(DataCleaningEncodingImputationMixin, DataCleaningTransformHandlersMixin):
    """Composite mixin bundling survey-cleaning transformation handlers."""
