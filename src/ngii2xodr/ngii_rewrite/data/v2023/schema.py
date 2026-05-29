"""2023.07 NGII rewrite schema."""

from __future__ import annotations

from ngii2xodr.ngii_rewrite.data.metadata import (
    ReciprocalReferenceRule,
    RoleFilter,
    Schema,
    validate_layer_types,
)
from ngii2xodr.ngii_rewrite.data.v2023.layers import LAYER_TYPES

validate_layer_types(LAYER_TYPES)

SCHEMA = Schema(
    version="2023.07",
    layer_types=LAYER_TYPES,
    role_filters=(
        RoleFilter("junction_link", "link", "link_type", ("1",)),
        RoleFilter("ordinary_link", "link", "link_type", ("4", "5", "6")),
        RoleFilter("pocket_link", "link", "lane_no", numeric_min=90.0),
        RoleFilter("uturn_marker", "lane_line", "kind", ("502",)),
        RoleFilter("junction_node", "node", "node_type", ("1",)),
    ),
    reciprocal_references=(
        ReciprocalReferenceRule("a2_link", "R_LinkID", "r_link_id", "L_LinkID", "l_link_id"),
        ReciprocalReferenceRule("a2_link", "L_LinkID", "l_link_id", "R_LinkID", "r_link_id"),
        ReciprocalReferenceRule(
            "c3_vehicleprotectionsafety", "Ref_ID", "ref_id", "Ref_ID", "ref_id"
        ),
    ),
)
