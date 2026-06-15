"""Knowledge-base layer: schema, the cached DatabaseManager, and the builder.

``KnowledgeBaseBuilder`` is exposed lazily (PEP 562) so importing this package —
or the lightweight :class:`DatabaseManager` — does not eagerly pull in pandas,
torch and sentence-transformers.
"""

from typing import TYPE_CHECKING

from lokma.knowledge.database_manager import DatabaseManager
from lokma.knowledge.schema import (
    CREATE_NUTRITION_TABLE,
    NUTRITION_COLUMNS,
    NUTRITION_TABLE,
)

if TYPE_CHECKING:
    from lokma.knowledge.builder import KnowledgeBaseBuilder

__all__ = [
    "CREATE_NUTRITION_TABLE",
    "DatabaseManager",
    "KnowledgeBaseBuilder",
    "NUTRITION_COLUMNS",
    "NUTRITION_TABLE",
]


def __getattr__(name: str):
    if name == "KnowledgeBaseBuilder":
        from lokma.knowledge.builder import KnowledgeBaseBuilder

        return KnowledgeBaseBuilder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
