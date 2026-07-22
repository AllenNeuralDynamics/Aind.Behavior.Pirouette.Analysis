"""Pirouette data: processing and analysis toolkit for chronic ephys and behavior data."""

from pirouette_data import ingestion, kinematics
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
    commutator_heading_estimate,
    heading_offset_from_ears,
    load_commutator_turns,
)

__version__ = "0.1.0"

__all__ = [
    "ingestion",
    "kinematics",
    "build_dataset",
    "flatten_pose_columns",
    "get_experiment_start_harp",
    "harp_to_datetime",
    "load_harp_seconds",
    "load_pose_h5",
    "parse_camera_and_timestamp",
    "append_commutator_heading",
    "commutator_heading_estimate",
    "heading_offset_from_ears",
    "load_commutator_turns",
]
