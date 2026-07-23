"""Pirouette data: processing and analysis toolkit for chronic ephys and behavior data."""

from pirouette_data import (
    behavior_classification,
    cli,
    ingestion,
    kinematics,
    processing,
    visualization_gui,
)
from pirouette_data.behavior_classification import (
    append_behavior_labels,
    classify_rest_movement,
    estimate_velocity_threshold,
)
from pirouette_data.ingestion import (
    build_dataset,
    flatten_pose_columns,
    get_experiment_start_harp,
    harp_to_datetime,
    load_harp_seconds,
    load_pose_h5,
    parse_camera_and_timestamp,
)
from pirouette_data.kinematics import (
    append_commutator_heading,
    append_ear_heading,
    append_ear_velocity,
    commutator_heading_estimate,
    ear_heading_estimate,
    ear_midpoint,
    ear_velocity_estimate,
    heading_offset_from_ears,
    load_commutator_turns,
)
from pirouette_data.processing import (
    ChamberScale,
    append_mm_columns,
    estimate_chamber_scale,
)

__version__ = "0.1.0"

__all__ = [
    "behavior_classification",
    "cli",
    "ingestion",
    "kinematics",
    "processing",
    "visualization_gui",
    "build_dataset",
    "flatten_pose_columns",
    "get_experiment_start_harp",
    "harp_to_datetime",
    "load_harp_seconds",
    "load_pose_h5",
    "parse_camera_and_timestamp",
    "append_commutator_heading",
    "append_ear_heading",
    "append_ear_velocity",
    "commutator_heading_estimate",
    "ear_heading_estimate",
    "ear_midpoint",
    "ear_velocity_estimate",
    "heading_offset_from_ears",
    "load_commutator_turns",
    "ChamberScale",
    "append_mm_columns",
    "estimate_chamber_scale",
    "append_behavior_labels",
    "classify_rest_movement",
    "estimate_velocity_threshold",
]
