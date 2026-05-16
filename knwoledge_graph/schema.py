
from neo4j_graphrag.experimental.components.schema import (
    NodeType as SchemaNodeType,
    PropertyType,
    RelationshipType as SchemaRelationshipType,
)
from datetime import date


_TODAY = date.today().strftime("%Y-%m-%d")

# ─────────────────────────────────────────────────────────────────────────────
# Strict schema definition
# ─────────────────────────────────────────────────────────────────────────────

NODE_TYPES = [
    # — People & organisations
    SchemaNodeType(label="Person",              properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Role",                properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Organization",        properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Location",            properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="country", type="STRING"), PropertyType(name="description", type="STRING")]),
    # — IT-specific technical nodes
    SchemaNodeType(label="ProgrammingLanguage", properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Framework",           properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Database",            properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="CloudPlatform",       properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="DevOpsTool",          properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Technology",          properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Domain",              properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Methodology",         properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    # — Skills & languages
    SchemaNodeType(label="Skill",               properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Language",            properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Proficiency",         properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    # — Education & credentials
    SchemaNodeType(label="Degree",              properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Certification",       properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Award",               properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    # — Projects & activities
    SchemaNodeType(label="Project",             properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="PersonalProject",     properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="ProjectUrl",          properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Activity",            properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    # — Contact & time
    SchemaNodeType(label="Contact",             properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="type", type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="DateRange",           properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
]

RELATIONSHIP_TYPES = [
    SchemaRelationshipType(
        label="WORKS_AT",
        description=(
            f"CURRENT employment (Person→Organization). Use ONLY when the position is ongoing as of today ({_TODAY}): "
            "the chunk says 'Present', 'Current', 'Now', 'ongoing', or has a start date with NO end date. "
            "If unsure, prefer WORKED_AT."
        ),
        properties=[PropertyType(name="description", type="STRING")],
    ),
    SchemaRelationshipType(
        label="WORKED_AT",
        description=(
            f"PAST employment (Person→Organization). Use when the end date is explicitly BEFORE today ({_TODAY}). "
            "Default choice when there is any doubt about whether the role is still active."
        ),
        properties=[PropertyType(name="description", type="STRING")],
    ),
    SchemaRelationshipType(label="HAS_ROLE",        description="Links a Person to a Role they hold or held (e.g. 'Software Engineer', 'Data Scientist').",        properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="ROLE_AT",         description="Links a Role to the Organization where it was or is held.",                                         properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="HAS_SKILL",       description="Links a Person to a Skill (technical or soft skill).",                                              properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="HAS_PROFICIENCY", description="Links a Skill or Language node to a Proficiency level (e.g. Beginner, Intermediate, Advanced, Fluent, Native).", properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="KNOWS_LANGUAGE",  description="A Person knows a natural Language (Italian, English, French, ...).",                                properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="HAS_CERTIFICATION", description="A Person holds a Certification.",                                                                 properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="HAS_AWARD",       description="A Person received an Award or recognition.",                                                        properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="STUDIED_AT",      description="A Person studied at an Organization (university, school, bootcamp).",                               properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="OBTAINED_DEGREE", description="A Person obtained a Degree (e.g. 'Bachelor of Science in Computer Science').",                      properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="GRANTED_BY",      description="A Certification, Award, or Degree was issued/granted by an Organization.",                          properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="HAS_CONTACT",     description="A Person has a Contact (email, phone, LinkedIn, GitHub, website). Set the `type` property on the Contact node.", properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="INVOLVED_IN",     description="A Person was involved in a Project or PersonalProject.",                                            properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="PARTICIPATED_IN", description="A Person participated in an Activity (volunteering, sport, association, event).",                   properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="LOCATED_IN",      description="A Person or Organization is located in a Location (city, country, region).",                        properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="USES",            description="A Role, Project, or PersonalProject uses a Technology, Framework, Database, or DevOpsTool. Use more specific relations (DEPLOYED_ON, APPLIES) when applicable.", properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="DEPLOYED_ON",     description="A Role, Project, or PersonalProject is deployed on or hosted by a CloudPlatform (e.g. AWS, Azure, GCP).", properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="SPECIALIZES_IN",  description="A Person or Role specializes in a Domain (e.g. Backend, Machine Learning, Data Engineering, NLP, DevOps, Frontend).", properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="APPLIES",         description="A Role or Project applies a Methodology or practice (e.g. Agile, Scrum, TDD, REST, Microservices, CI/CD).", properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="DURING",          description="Links a Role, Project, PersonalProject, Certification, Degree, Award, or Activity to its DateRange (e.g. '2020–2023', 'Jan 2021 – Present').", properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="HAS_URL",         description="Links a Project or PersonalProject to a ProjectUrl.",                                               properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="PART_OF",         description="Links a Project to the Organization it was developed for or within.",                               properties=[PropertyType(name="description", type="STRING")]),
]

PATTERNS = [
    # Person — employment
    ("Person",          "WORKS_AT",         "Organization"),
    ("Person",          "WORKED_AT",        "Organization"),
    ("Person",          "HAS_ROLE",         "Role"),
    # Role
    ("Role",            "ROLE_AT",          "Organization"),
    ("Role",            "WORKS_AT",         "Organization"),
    ("Role",            "SPECIALIZES_IN",   "Domain"),
    ("Role",            "USES",             "ProgrammingLanguage"),
    ("Role",            "USES",             "Framework"),
    ("Role",            "USES",             "Database"),
    ("Role",            "USES",             "DevOpsTool"),
    ("Role",            "USES",             "Technology"),
    ("Role",            "DEPLOYED_ON",      "CloudPlatform"),
    ("Role",            "APPLIES",          "Methodology"),
    ("Role",            "DURING",           "DateRange"),
    # Person — IT specialisation
    ("Person",          "SPECIALIZES_IN",   "Domain"),
    # Person — skills & languages
    ("Person",          "HAS_SKILL",        "Skill"),
    ("Person",          "KNOWS_LANGUAGE",   "Language"),
    ("Skill",           "HAS_PROFICIENCY",  "Proficiency"),
    ("Language",        "HAS_PROFICIENCY",  "Proficiency"),
    # Person — education
    ("Person",          "STUDIED_AT",       "Organization"),
    ("Person",          "OBTAINED_DEGREE",  "Degree"),
    ("Degree",          "GRANTED_BY",       "Organization"),
    ("Degree",          "DURING",           "DateRange"),
    # Person — certifications
    ("Person",          "HAS_CERTIFICATION","Certification"),
    ("Certification",   "GRANTED_BY",       "Organization"),
    ("Certification",   "DURING",           "DateRange"),
    # Person — awards
    ("Person",          "HAS_AWARD",        "Award"),
    ("Award",           "GRANTED_BY",       "Organization"),
    ("Award",           "DURING",           "DateRange"),
    # Person — contacts & location
    ("Person",          "HAS_CONTACT",      "Contact"),
    ("Person",          "LOCATED_IN",       "Location"),
    ("Organization",    "LOCATED_IN",       "Location"),
    # Person — activities
    ("Person",          "PARTICIPATED_IN",  "Activity"),
    ("Activity",        "DURING",           "DateRange"),
    # Projects
    ("Person",          "INVOLVED_IN",      "Project"),
    ("Person",          "INVOLVED_IN",      "PersonalProject"),
    ("Project",         "USES",             "ProgrammingLanguage"),
    ("Project",         "USES",             "Framework"),
    ("Project",         "USES",             "Database"),
    ("Project",         "USES",             "DevOpsTool"),
    ("Project",         "USES",             "Technology"),
    ("Project",         "DEPLOYED_ON",      "CloudPlatform"),
    ("Project",         "APPLIES",          "Methodology"),
    ("Project",         "DURING",           "DateRange"),
    ("Project",         "HAS_URL",          "ProjectUrl"),
    ("Project",         "PART_OF",          "Organization"),
    ("PersonalProject", "USES",             "ProgrammingLanguage"),
    ("PersonalProject", "USES",             "Framework"),
    ("PersonalProject", "USES",             "Database"),
    ("PersonalProject", "USES",             "DevOpsTool"),
    ("PersonalProject", "USES",             "Technology"),
    ("PersonalProject", "DEPLOYED_ON",      "CloudPlatform"),
    ("PersonalProject", "APPLIES",          "Methodology"),
    ("PersonalProject", "DURING",           "DateRange"),
    ("PersonalProject", "HAS_URL",          "ProjectUrl"),
]

# ─────────────────────────────────────────────────────────────────────────────
# GitHub README schema — nodes, relationships and allowed patterns
# ─────────────────────────────────────────────────────────────────────────────

GITHUB_NODE_TYPES = [
    SchemaNodeType(label="Person",              properties=[PropertyType(name="name",    type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="PersonalProject",    properties=[PropertyType(name="name", type="STRING"), PropertyType(name="url", type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Technology",         properties=[PropertyType(name="name", type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="ProgrammingLanguage",properties=[PropertyType(name="name", type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Framework",          properties=[PropertyType(name="name", type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Database",           properties=[PropertyType(name="name", type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="CloudPlatform",      properties=[PropertyType(name="name", type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="DevOpsTool",         properties=[PropertyType(name="name", type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Concept",            properties=[PropertyType(name="name", type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Topic",              properties=[PropertyType(name="name", type="STRING"), PropertyType(name="description", type="STRING")]),
    SchemaNodeType(label="Organization",       properties=[PropertyType(name="name", type="STRING"), PropertyType(name="description", type="STRING")]),
]

GITHUB_RELATIONSHIP_TYPES = [
    SchemaRelationshipType(label="CONTRIBUTED_TO",  description="Person contributed to a PersonalProject.",                       properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="USES",            description="PersonalProject uses a Technology or Framework.",                                    properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="BUILT_WITH",      description="PersonalProject is built with a Technology, Framework, or ProgrammingLanguage.",    properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="WRITTEN_IN",      description="PersonalProject is primarily written in a ProgrammingLanguage.",                    properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="DEPENDS_ON",      description="PersonalProject depends on a Technology or Framework.",                             properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="INTEGRATES_WITH", description="PersonalProject integrates with an external Technology, API, or Organization.",     properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="DEPLOYED_ON",     description="PersonalProject is deployed on or hosted by a CloudPlatform.",                      properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="HOSTED_ON",       description="PersonalProject is hosted on a Technology or Organization platform.",               properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="IMPLEMENTS",      description="PersonalProject implements a Concept, pattern, or algorithm.",                      properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="SOLVES",          description="PersonalProject solves a problem described by a Concept.",                          properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="DEMONSTRATES",    description="PersonalProject demonstrates a Concept or Skill.",                                  properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="HAS_TOPIC",       description="PersonalProject is tagged with a Topic.",                                           properties=[PropertyType(name="description", type="STRING")]),
    SchemaRelationshipType(label="RELATED_TO",      description="A Technology or Concept is semantically related to another.",                       properties=[PropertyType(name="description", type="STRING")]),
]

GITHUB_PATTERNS = [
    ("PersonalProject", "USES",            "Technology"),
    ("PersonalProject", "USES",            "Framework"),
    ("PersonalProject", "USES",            "Database"),
    ("PersonalProject", "USES",            "DevOpsTool"),
    ("PersonalProject", "BUILT_WITH",      "Technology"),
    ("PersonalProject", "BUILT_WITH",      "Framework"),
    ("PersonalProject", "BUILT_WITH",      "ProgrammingLanguage"),
    ("PersonalProject", "WRITTEN_IN",      "ProgrammingLanguage"),
    ("PersonalProject", "DEPENDS_ON",      "Technology"),
    ("PersonalProject", "DEPENDS_ON",      "Framework"),
    ("PersonalProject", "INTEGRATES_WITH", "Technology"),
    ("PersonalProject", "INTEGRATES_WITH", "Organization"),
    ("PersonalProject", "DEPLOYED_ON",     "CloudPlatform"),
    ("PersonalProject", "HOSTED_ON",       "Technology"),
    ("PersonalProject", "HOSTED_ON",       "Organization"),
    ("PersonalProject", "IMPLEMENTS",      "Concept"),
    ("PersonalProject", "SOLVES",          "Concept"),
    ("PersonalProject", "DEMONSTRATES",    "Concept"),
    ("PersonalProject", "HAS_TOPIC",       "Topic"),
    ("Technology",      "RELATED_TO",      "Technology"),
    ("Concept",         "RELATED_TO",      "Concept"),
    ("Person",         "CONTRIBUTED_TO",  "PersonalProject"),
]
