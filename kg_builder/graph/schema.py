"""Per-source graph schema definitions.

Each :class:`SourceSchema` constrains the entities/relationships an LLM is
allowed to extract from a given document kind. Tight schemas keep the graph
clean and prevent the cover letter from inventing ``PersonalProject`` nodes
that should come from GitHub READMEs.

We deliberately keep ``strict=False``: the LLM should pick the best match from
the *suggested* vocabulary but is allowed to mint a new label/relationship if
nothing fits — the canonicalisation pass cleans up the rest. ``strict=True``
in practice drops too much information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class SourceSchema:
    name: str
    allowed_nodes: Sequence[str]
    allowed_relationships: Sequence[str] = ()
    strict: bool = False
    instructions: str = ""


# ---------------------------------------------------------------------------
# CV / Cover Letter
# ---------------------------------------------------------------------------

# Note: ``PersonalProject`` / ``Project`` are intentionally absent — projects
# come from the GitHub source.
_CV_NODES = (
    "Person",
    "Role",
    "Skill",
    "Technology",
    "ProgrammingLanguage",
    "Organization",
    "Language",
    "Concept",
    "Contact",
    "Certification",
    "Activity",
    "DateRange",
    "Location",
    "Proficiency",
)

# Relationships are listed grouped by domain so the LLM has a clear menu.
# Direction conventions: subject → object (e.g. Person -[WORKED_AS]-> Role).
_CV_RELATIONSHIPS = (
    # Career
    "WORKED_AS",          # Person -> Role
    "WORKED_FOR",         # Person -> Organization
    "AT",                 # Role -> Organization
    "REPORTED_TO",        # Person -> Person/Role
    "LED",                # Person -> Activity / Concept
    "RESPONSIBLE_FOR",    # Person/Role -> Concept / Activity
    "COLLABORATED_WITH",  # Person -> Person / Organization
    # Education
    "STUDIED_AT",         # Person -> Organization
    "GRADUATED_FROM",     # Person -> Organization
    "EARNED",             # Person -> Certification
    "HOLDS",              # Person -> Certification
    "ISSUED_BY",          # Certification -> Organization
    # Skills & technologies
    "HAS_SKILL",          # Person -> Skill
    "PROFICIENT_IN",      # Person -> Skill / Technology
    "KNOWS",              # Person -> Technology / ProgrammingLanguage
    "USED",               # Role -> Technology  (during a job)
    "INVOLVED",           # Role -> Skill
    "FOCUSED_ON",         # Role -> Concept
    "RELATED_TO",         # Skill -> Technology / Concept
    # Languages
    "SPEAKS",             # Person -> Language
    # Activities & projects (work-context, not personal repos)
    "PARTICIPATED_IN",    # Person -> Activity
    "ORGANIZED_BY",       # Activity -> Organization
    # Locations
    "LIVES_IN",           # Person -> Location
    "BORN_IN",            # Person -> Location
    "LOCATED_IN",         # Organization / Activity -> Location
    # Time
    "DURING",             # Role / Activity -> DateRange
    "FROM",               # DateRange -> (date marker)
    "UNTIL",              # DateRange -> (date marker)
    # Contact
    "CONTACT_VIA",        # Person -> Contact
    # Proficiency level annotation
    "AT_LEVEL",           # Skill / Technology / Language -> Proficiency
)

CV_SCHEMA = SourceSchema(
    name="cv",
    allowed_nodes=_CV_NODES,
    allowed_relationships=_CV_RELATIONSHIPS,
    strict=False,
    instructions=(
        "You are extracting entities from a CV or cover letter. "
        "**Do NOT extract any PersonalProject, Project or ProjectUrl nodes.** "
        "Personal projects are managed by a separate pipeline that ingests "
        "GitHub READMEs directly. "
        "Job titles ('Software Engineer', 'AI Engineer', 'Consultant') are "
        "ALWAYS Role nodes — never Person. Companies/universities/schools are "
        "Organization nodes. Cities/countries are Location nodes. "
        "Use canonical Title Case ids ('Python', 'Microsoft Azure') and pick a "
        "relationship from the suggested vocabulary whenever one fits — only "
        "mint a new one if nothing in the list matches the meaning."
    ),
)


# ---------------------------------------------------------------------------
# GitHub README
# ---------------------------------------------------------------------------

_GITHUB_NODES = (
    "PersonalProject",
    "Technology",
    "ProgrammingLanguage",
    "Concept",
    "Skill",
    "Topic",
    "Organization",  # e.g. cloud providers, integrations
)

_GITHUB_RELATIONSHIPS = (
    # Tech stack
    "USES",              # PersonalProject -> Technology
    "BUILT_WITH",        # PersonalProject -> Technology / ProgrammingLanguage
    "WRITTEN_IN",        # PersonalProject -> ProgrammingLanguage
    "DEPENDS_ON",        # PersonalProject -> Technology
    "INTEGRATES_WITH",   # PersonalProject -> Technology / Organization
    "DEPLOYED_ON",       # PersonalProject -> Technology / Organization
    "HOSTED_ON",         # PersonalProject -> Technology / Organization
    # Functional / semantic
    "IMPLEMENTS",        # PersonalProject -> Concept
    "SOLVES",            # PersonalProject -> Concept
    "SUPPORTS",          # PersonalProject -> Concept / Technology
    "EXPOSES",           # PersonalProject -> Concept (e.g. an API surface)
    "DEMONSTRATES",      # PersonalProject -> Skill / Concept
    # Topics / categorisation
    "HAS_TOPIC",         # PersonalProject -> Topic
    "BELONGS_TO",        # PersonalProject -> Concept (domain area)
    # Tech-to-tech relationships when described in the README
    "PART_OF",           # Technology -> Technology
    "ALTERNATIVE_TO",    # Technology -> Technology
    "RELATED_TO",        # Technology / Concept -> Technology / Concept
)

GITHUB_SCHEMA = SourceSchema(
    name="github_readme",
    allowed_nodes=_GITHUB_NODES,
    allowed_relationships=_GITHUB_RELATIONSHIPS,
    strict=False,
    instructions=(
        "You are extracting entities from a GitHub README of a personal "
        "project. The PersonalProject node has already been created by the "
        "pipeline (do not create new ones); instead extract the technologies, "
        "programming languages, concepts and topics the project uses, and "
        "connect them to the existing project node. "
        "Prefer relationships from the suggested vocabulary: USES / BUILT_WITH "
        "/ WRITTEN_IN / DEPENDS_ON / INTEGRATES_WITH / DEPLOYED_ON / "
        "IMPLEMENTS / SOLVES / SUPPORTS / DEMONSTRATES / HAS_TOPIC. "
        "Do NOT create a Person node — the project owner is managed elsewhere. "
        "Use canonical Title Case ids."
    ),
)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def schema_for_kind(kind: str) -> SourceSchema:
    """Map a :class:`SourceDocument.kind` to a :class:`SourceSchema`."""
    if kind == "github_readme":
        return GITHUB_SCHEMA
    return CV_SCHEMA
