"""2025.12 NGII rewrite schema."""

from __future__ import annotations

from ngii2xodr.ngii_rewrite.data.metadata import (
    ReciprocalReferenceRule,
    RoleFilter,
    RoleKey,
    Schema,
    validate_layer_types,
)
from ngii2xodr.ngii_rewrite.data.v2025.layers import LAYER_TYPES

validate_layer_types(LAYER_TYPES)

SCHEMA = Schema(
    version="2025.12",
    layer_types=LAYER_TYPES,
    role_filters=(
        RoleFilter("junction_link", "link", "link_type", ("100", "102")),
        RoleFilter("ordinary_link", "link", "link_type", ("200", "201", "300")),
        RoleFilter("pocket_link", "link", "turn", ("1", "3")),
        RoleFilter("uturn_link", "link", "turn", ("3",)),
        RoleFilter("uturn_marker", "lane_line", "line_kind", ("502",)),
        RoleFilter(
            "junction_node",
            "node",
            values=("100",),
            attr_names=("node_type1", "node_type2", "node_type3"),
        ),
    ),
    role_keys=(RoleKey("node_group", "node", "group_id"),),
    reciprocal_references=(
        ReciprocalReferenceRule("nt2_link", "R_LinkID", "r_link_id", "L_LinkID", "l_link_id"),
        ReciprocalReferenceRule("nt2_link", "L_LinkID", "l_link_id", "R_LinkID", "r_link_id"),
    ),
)
