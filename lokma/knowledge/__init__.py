"""Knowledge-base layer: taxonomy, normalized schema, resolver, cached manager.

``KnowledgeBaseBuilder`` is exposed lazily (PEP 562) so importing this package —
or the lightweight :class:`DatabaseManager` / :class:`EntityResolver` — does not
eagerly pull in pandas or sentence-transformers.
"""

from typing import TYPE_CHECKING

from lokma.knowledge.database_manager import DatabaseManager
from lokma.knowledge.entity_resolver import EntityResolver, ResolvedFood
from lokma.knowledge.schema import NUTRITION_COLUMNS, SCHEMA_VERSION

if TYPE_CHECKING:
    from lokma.knowledge.builder import KnowledgeBaseBuilder

__all__ = [
    "DatabaseManager",
    "EntityResolver",
    "KnowledgeBaseBuilder",
    "NUTRITION_COLUMNS",
    "ResolvedFood",
    "SCHEMA_VERSION",
]


def __getattr__(name: str):
    if name == "KnowledgeBaseBuilder":
        from lokma.knowledge.builder import KnowledgeBaseBuilder

        return KnowledgeBaseBuilder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
