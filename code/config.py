"""Shared configuration for the realtime demo and evaluation script."""

# Geometric thresholds
EAR_THRESHOLD = 0.21
MAR_THRESHOLD = 0.60
TILT_THRESHOLD = 15.0

# Time for which a sign must remain continuous before an alert is raised.
# This is the single source of truth; do not redefine it as a frame count.
EAR_CONSEC_SECONDS = 1.0          # equivalent to 30 frames at 30 FPS
MAR_CONSEC_SECONDS = 0.5          # equivalent to 15 frames at 30 FPS
TILT_CONSEC_SECONDS = 2.0 / 3.0   # equivalent to 20 frames at 30 FPS

# Used only to explain/compare the legacy configuration in the report.
REFERENCE_FPS = 30.0
