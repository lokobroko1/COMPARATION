#!/usr/bin/env python3
"""
SIZE COMPARISON: FROM QUARK TO OBSERVABLE UNIVERSE
Full HD 1920x1080 · ~3 min · Python + NumPy + Pillow + ffmpeg

Dependencies:
    pip install pillow imageio imageio-ffmpeg scipy tqdm numpy

Usage:
    python size_comparison_3d.py

Output: size_comparison_3d.mp4
"""

# ---------------------------------------------------------------------------
# Section 1: IMPORTS
# ---------------------------------------------------------------------------
import subprocess
import sys
import os
import shutil
import math
import time
import gc
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import imageio
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# Section 2: CONFIGURATION
# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 1920, 1080
FPS_RENDER = 15
# FPS_OUTPUT must be an integer multiple of FPS_RENDER (each rendered frame is duplicated).
FPS_OUTPUT = 30
assert FPS_OUTPUT % FPS_RENDER == 0, (
    f"FPS_OUTPUT ({FPS_OUTPUT}) must be a multiple of FPS_RENDER ({FPS_RENDER})"
)
_FRAME_REPEAT = FPS_OUTPUT // FPS_RENDER  # how many times each rendered frame is written
BG_COLOR = (4, 3, 18)

SEC_APPEAR = 1.5
SEC_HOLD = 5.5
SEC_TRANSITION = 2.5

F_APPEAR = int(SEC_APPEAR * FPS_RENDER)
F_HOLD = int(SEC_HOLD * FPS_RENDER)
F_TRANSITION = int(SEC_TRANSITION * FPS_RENDER)
F_PER_OBJ = F_APPEAR + F_HOLD + F_TRANSITION

# Raymarching constants (tunable here alongside the other render settings)
# MAX_STEPS: sphere-marching iteration budget. Higher = more accurate at complex geometry;
#            lower = faster renders. 80 is a good balance for unit-scale SDF scenes.
MAX_STEPS = 80
# MAX_DIST: rays that travel beyond this distance are considered misses (background).
MAX_DIST = 20.0
# SURF_DIST: a ray is considered to have hit a surface when d(p) < SURF_DIST.
#            Too small → slow convergence; too large → fat surfaces / missed thin details.
SURF_DIST = 0.002

# ---------------------------------------------------------------------------
# Section 3: OBJECT DATA
# ---------------------------------------------------------------------------
OBJECTS = [
    {
        'name': 'Up Quark',
        'size_m': 1e-18,
        'color': (255, 80, 80),
        'glow_color': (200, 20, 20),
        'category': 'SUBATOMIC PARTICLE',
        'facts': [
            'One of the smallest known constituents of matter',
            'Quarks are confined — never found alone (color confinement)',
            'Charge: +2/3 e  |  Mass: ~2.3 MeV/c²',
        ],
        'si_label': '~10⁻¹⁸ m  (1 attometer)',
        'us_label': '~3.9 × 10⁻¹⁷ inches',
        'human_ref': False,
        'earth_ref': False,
    },
    {
        'name': 'Proton',
        'size_m': 1.7e-15,
        'color': (255, 140, 60),
        'glow_color': (200, 80, 10),
        'category': 'SUBATOMIC PARTICLE',
        'facts': [
            'Made of 2 up quarks and 1 down quark',
            'Radius ~0.85 femtometers; proton charge radius debate ongoing',
            '~1836× heavier than the electron',
        ],
        'si_label': '1.7 × 10⁻¹⁵ m  (1.7 femtometers)',
        'us_label': '6.7 × 10⁻¹⁴ inches',
        'human_ref': False,
        'earth_ref': False,
    },
    {
        'name': 'Hydrogen Atom',
        'size_m': 1.2e-10,
        'color': (80, 200, 255),
        'glow_color': (20, 140, 220),
        'category': 'ATOM',
        'facts': [
            'Smallest and most abundant element in the universe',
            'Bohr radius ~53 pm; quantum cloud extends further',
            '~100,000× larger than its nucleus',
        ],
        'si_label': '1.2 × 10⁻¹⁰ m  (0.12 nm)',
        'us_label': '4.7 × 10⁻⁹ inches',
        'human_ref': False,
        'earth_ref': False,
    },
    {
        'name': 'DNA Molecule',
        'size_m': 2.5e-9,
        'color': (80, 220, 150),
        'glow_color': (20, 160, 90),
        'category': 'MOLECULE',
        'facts': [
            'Double helix diameter ~2.5 nm; encodes all life',
            'Human DNA stretched out would reach ~2 meters',
            '~6 billion base pairs in human genome',
        ],
        'si_label': '2.5 × 10⁻⁹ m  (2.5 nm)',
        'us_label': '9.8 × 10⁻⁸ inches',
        'human_ref': False,
        'earth_ref': False,
    },
    {
        'name': 'SARS-CoV-2 Virus',
        'size_m': 1.2e-7,
        'color': (220, 80, 220),
        'glow_color': (160, 20, 160),
        'category': 'VIRUS',
        'facts': [
            'Diameter ~100–140 nm with distinctive spike proteins',
            'Spike proteins bind to ACE2 receptors in human cells',
            'RNA genome ~30,000 nucleotides long',
        ],
        'si_label': '1.2 × 10⁻⁷ m  (120 nm)',
        'us_label': '4.7 × 10⁻⁶ inches',
        'human_ref': False,
        'earth_ref': False,
    },
    {
        'name': 'Red Blood Cell',
        'size_m': 8e-6,
        'color': (220, 40, 40),
        'glow_color': (160, 10, 10),
        'category': 'CELL',
        'facts': [
            'Biconcave disc ~6–8 µm in diameter',
            'No nucleus — maximises oxygen-carrying haemoglobin',
            '~25 trillion red blood cells in the human body',
        ],
        'si_label': '8 × 10⁻⁶ m  (8 micrometers)',
        'us_label': '3.1 × 10⁻⁴ inches',
        'human_ref': False,
        'earth_ref': False,
    },
    {
        'name': 'Ant',
        'size_m': 2e-3,
        'color': (160, 100, 40),
        'glow_color': (100, 60, 10),
        'category': 'INSECT',
        'facts': [
            'Can lift 10–50× its own body weight',
            'About 20 quadrillion ants live on Earth',
            'Highly eusocial; colonies can have millions of workers',
        ],
        'si_label': '2 × 10⁻³ m  (2 mm)',
        'us_label': '0.08 inches',
        'human_ref': False,
        'earth_ref': False,
    },
    {
        'name': 'Human',
        'size_m': 1.75,
        'color': (220, 170, 130),
        'glow_color': (160, 110, 70),
        'category': 'ORGANISM',
        'facts': [
            'Average height ~1.75 m (5 ft 9 in)',
            'Body contains ~37 trillion cells',
            '~8 billion humans currently on Earth',
        ],
        'si_label': '1.75 m',
        'us_label': '5 ft 9 in  (69 inches)',
        'human_ref': True,
        'earth_ref': False,
    },
    {
        'name': 'Eiffel Tower',
        'size_m': 324.0,
        'color': (160, 150, 130),
        'glow_color': (100, 90, 70),
        'category': 'STRUCTURE',
        'facts': [
            'Height 324 m including antenna; completed 1889',
            'Made of ~7,300 tonnes of wrought iron',
            'Visited by ~7 million people per year',
        ],
        'si_label': '324 m',
        'us_label': '1,063 feet  (0.2 miles)',
        'human_ref': False,
        'earth_ref': False,
    },
    {
        'name': 'Mount Everest',
        'size_m': 8848.86,
        'color': (200, 200, 210),
        'glow_color': (140, 140, 160),
        'category': 'MOUNTAIN',
        'facts': [
            'Highest point on Earth at 8,848.86 m above sea level',
            'Part of the Himalayan range; still rising ~4 mm/year',
            'First summited by Hillary & Tenzing on 29 May 1953',
        ],
        'si_label': '8,848.86 m  (8.85 km)',
        'us_label': '29,032 feet  (5.5 miles)',
        'human_ref': False,
        'earth_ref': False,
    },
    {
        'name': 'Earth',
        'size_m': 1.274e7,
        'color': (60, 130, 200),
        'glow_color': (30, 80, 150),
        'category': 'PLANET',
        'facts': [
            'Mean diameter 12,742 km; slightly oblate at poles',
            'Only known planet harboring life',
            'Surface is 71% water; 1 Moon in orbit',
        ],
        'si_label': '12,742 km',
        'us_label': '7,918 miles',
        'human_ref': False,
        'earth_ref': True,
    },
    {
        'name': 'Jupiter',
        'size_m': 1.4e8,
        'color': (210, 160, 100),
        'glow_color': (160, 110, 60),
        'category': 'PLANET',
        'facts': [
            'Largest planet; diameter ~11× Earth\'s',
            'Great Red Spot is a storm lasting >350 years',
            '95 known moons including Ganymede, Europa, Io',
        ],
        'si_label': '139,820 km',
        'us_label': '86,881 miles',
        'human_ref': False,
        'earth_ref': False,
    },
    {
        'name': 'The Sun',
        'size_m': 1.39e9,
        'color': (255, 220, 60),
        'glow_color': (220, 160, 10),
        'category': 'STAR',
        'facts': [
            'Diameter ~1.39 million km; ~109× Earth\'s diameter',
            'Contains 99.86% of the Solar System\'s total mass',
            'Core temperature ~15 million °C; fuses 600 Mt H/s',
        ],
        'si_label': '1,392,700 km',
        'us_label': '865,370 miles',
        'human_ref': False,
        'earth_ref': False,
    },
    {
        'name': 'Solar System',
        'size_m': 9e12,
        'color': (180, 180, 220),
        'glow_color': (120, 120, 160),
        'category': 'PLANETARY SYSTEM',
        'facts': [
            'Diameter to heliopause ~9 trillion km (~60 AU)',
            'Orbits take 0.24 y (Mercury) to 165 y (Neptune)',
            'Oort Cloud extends to ~100,000 AU',
        ],
        'si_label': '~9 × 10¹² m  (9 trillion km)',
        'us_label': '~5.6 × 10¹² miles',
        'human_ref': False,
        'earth_ref': False,
    },
    {
        'name': 'Light Year',
        'size_m': 9.461e15,
        'color': (100, 180, 255),
        'glow_color': (60, 120, 200),
        'category': 'ASTRONOMICAL DISTANCE',
        'facts': [
            'Distance light travels in one year: ~9.46 × 10¹⁵ m',
            '~63,241 AU; ~5.88 trillion miles',
            'Nearest star (Proxima Centauri) is 4.24 light years away',
        ],
        'si_label': '9.461 × 10¹⁵ m',
        'us_label': '5.879 × 10¹² miles',
        'human_ref': False,
        'earth_ref': False,
    },
    {
        'name': 'Milky Way Galaxy',
        'size_m': 9.461e20,
        'color': (200, 180, 255),
        'glow_color': (140, 120, 200),
        'category': 'GALAXY',
        'facts': [
            'Diameter ~100,000 light years; barred spiral galaxy',
            'Contains 100–400 billion stars',
            'Our Solar System orbits the centre every ~225–250 Myr',
        ],
        'si_label': '9.461 × 10²⁰ m  (100,000 ly)',
        'us_label': '5.88 × 10¹⁷ miles',
        'human_ref': False,
        'earth_ref': False,
    },
    {
        'name': 'Local Group',
        'size_m': 3.086e22,
        'color': (160, 200, 255),
        'glow_color': (100, 140, 200),
        'category': 'GALAXY GROUP',
        'facts': [
            'Diameter ~3 million light years; ~50+ galaxies',
            'Dominated by Milky Way and Andromeda (M31)',
            'Andromeda is approaching at ~110 km/s; merge in ~4.5 Gyr',
        ],
        'si_label': '3.086 × 10²² m  (~3 Mly)',
        'us_label': '~1.92 × 10¹⁹ miles',
        'human_ref': False,
        'earth_ref': False,
    },
    {
        'name': 'Observable Universe',
        'size_m': 8.8e26,
        'color': (180, 160, 255),
        'glow_color': (120, 100, 220),
        'category': 'COSMOS',
        'facts': [
            'Diameter ~93 billion light years due to expansion',
            'Contains ~2 trillion galaxies and ~10⁸⁰ atoms',
            'Age ~13.8 billion years since the Big Bang',
        ],
        'si_label': '8.8 × 10²⁶ m  (~93 billion ly)',
        'us_label': '~5.5 × 10²³ miles',
        'human_ref': False,
        'earth_ref': False,
    },
]

# ---------------------------------------------------------------------------
# Section 4: MATH UTILITIES
# ---------------------------------------------------------------------------

def ease_in_out_cubic(t):
    """Cubic ease in-out."""
    if t < 0.5:
        return 4 * t * t * t
    else:
        p = 2 * t - 2
        return 1.0 - p * p * p / 2


def ease_out_bounce(t):
    """Bounce easing for appear animation."""
    n1, d1 = 7.5625, 2.75
    if t < 1 / d1:
        return n1 * t * t
    elif t < 2 / d1:
        t -= 1.5 / d1
        return n1 * t * t + 0.75
    elif t < 2.5 / d1:
        t -= 2.25 / d1
        return n1 * t * t + 0.9375
    else:
        t -= 2.625 / d1
        return n1 * t * t + 0.984375


def rot_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def rot_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def apply_rotation(p, R):
    return p @ R.T

# ---------------------------------------------------------------------------
# Section 5: SDF PRIMITIVES
# ---------------------------------------------------------------------------

def sdf_sphere(p, center, radius):
    center = np.asarray(center, dtype=np.float32)
    d = p - center
    return np.sqrt(np.sum(d * d, axis=-1)) - radius


def sdf_box(p, center, half_extents):
    center = np.asarray(center, dtype=np.float32)
    half_extents = np.asarray(half_extents, dtype=np.float32)
    q = np.abs(p - center) - half_extents
    return (np.sqrt(np.sum(np.maximum(q, 0.0) ** 2, axis=-1))
            + np.minimum(np.max(q, axis=-1), 0.0))


def sdf_capsule(p, a, b, radius):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    pa = p - a
    ba = b - a
    h = np.clip(np.sum(pa * ba, axis=-1) / (np.sum(ba * ba) + 1e-10), 0.0, 1.0)
    d = pa - ba * h[..., np.newaxis]
    return np.sqrt(np.sum(d * d, axis=-1)) - radius


def sdf_cylinder(p, a, b, radius):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    ba = b - a
    pa = p - a
    baba = np.dot(ba, ba) + 1e-10
    paba = np.sum(pa * ba, axis=-1) / baba
    x = np.sqrt(np.sum((pa - ba * paba[..., np.newaxis]) ** 2, axis=-1) + 1e-10) - radius
    y = (np.abs(paba - 0.5) - 0.5) * np.sqrt(baba)
    x2 = x * x
    y2 = y * y
    d = np.where(
        np.maximum(x, y) < 0,
        -np.minimum(np.abs(x), np.abs(y)),
        np.where(x > 0, np.where(y > 0, np.sqrt(x2 + y2), x), y),
    )
    return d


def sdf_torus(p, center, R, r):
    center = np.asarray(center, dtype=np.float32)
    q = p - center
    xz = np.stack([q[..., 0], q[..., 2]], axis=-1)
    xz_len = np.sqrt(np.sum(xz ** 2, axis=-1)) - R
    d = np.stack([xz_len, q[..., 1]], axis=-1)
    return np.sqrt(np.sum(d ** 2, axis=-1)) - r


def sdf_ellipsoid(p, center, radii):
    center = np.asarray(center, dtype=np.float32)
    radii = np.asarray(radii, dtype=np.float32)
    q = (p - center) / (radii + 1e-10)
    ql = np.sqrt(np.sum(q ** 2, axis=-1))
    min_r = np.min(radii)
    return (ql - 1.0) * min_r


def sdf_cone(p, tip, base_center, radius):
    tip = np.asarray(tip, dtype=np.float32)
    base_center = np.asarray(base_center, dtype=np.float32)
    d = base_center - tip
    h = np.sqrt(np.sum(d ** 2)) + 1e-10
    axis = d / h
    pa = p - tip
    t = np.clip(np.sum(pa * axis, axis=-1), 0.0, h)
    closest = tip + axis * t[..., np.newaxis]
    radial_r = radius * t / h
    q = p - closest
    return np.sqrt(np.sum(q ** 2, axis=-1) + 1e-10) - radial_r

# ---------------------------------------------------------------------------
# Section 6: SDF BOOLEAN OPERATIONS
# ---------------------------------------------------------------------------

def sdf_union(d1, d2):
    return np.minimum(d1, d2)


def sdf_subtract(d1, d2):
    return np.maximum(d1, -d2)


def sdf_intersect(d1, d2):
    return np.maximum(d1, d2)


def sdf_smooth_union(d1, d2, k=0.1):
    h = np.clip(0.5 + 0.5 * (d2 - d1) / (k + 1e-10), 0.0, 1.0)
    return d2 * (1 - h) + d1 * h - k * h * (1 - h)

# ---------------------------------------------------------------------------
# Section 7: SDF OBJECT MODELS
# ---------------------------------------------------------------------------

def sdf_quark(p, t):
    """Vibrating energy point with pulsing sphere and ring structures."""
    pulse = 0.15 + 0.03 * np.sin(t * np.pi * 6)
    center = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    core = sdf_sphere(p, center, pulse)

    # Ring 1: in XZ plane (torus around Y axis)
    ring1 = sdf_torus(p, center, 0.35, 0.04)

    # Ring 2: tilted 60 degrees around X axis
    R60 = rot_x(np.radians(60.0))
    p_r60 = apply_rotation(p, R60)
    ring2 = sdf_torus(p_r60, center, 0.35, 0.04)

    # Ring 3: tilted 120 degrees around X axis
    R120 = rot_x(np.radians(120.0))
    p_r120 = apply_rotation(p, R120)
    ring3 = sdf_torus(p_r120, center, 0.35, 0.04)

    # Outer glow sphere
    glow = sdf_sphere(p, center, 0.55)

    d = sdf_smooth_union(core, ring1, k=0.08)
    d = sdf_smooth_union(d, ring2, k=0.08)
    d = sdf_smooth_union(d, ring3, k=0.08)
    d = sdf_smooth_union(d, glow, k=0.12)
    return d


def sdf_proton(p, t):
    """3 quarks connected by gluon tubes inside binding sphere."""
    q1_pos = np.array([0.3, 0.0, 0.0], dtype=np.float32)
    q2_pos = np.array([-0.15, 0.26, 0.0], dtype=np.float32)
    q3_pos = np.array([-0.15, -0.26, 0.0], dtype=np.float32)

    q1 = sdf_sphere(p, q1_pos, 0.18)
    q2 = sdf_sphere(p, q2_pos, 0.18)
    q3 = sdf_sphere(p, q3_pos, 0.18)

    # Gluon tubes connecting quarks
    g12 = sdf_cylinder(p, q1_pos, q2_pos, 0.05)
    g23 = sdf_cylinder(p, q2_pos, q3_pos, 0.05)
    g31 = sdf_cylinder(p, q3_pos, q1_pos, 0.05)

    # Outer binding sphere
    binding = sdf_sphere(p, np.array([0.0, 0.0, 0.0], dtype=np.float32), 0.55)

    d = sdf_smooth_union(q1, q2, k=0.06)
    d = sdf_smooth_union(d, q3, k=0.06)
    d = sdf_smooth_union(d, g12, k=0.04)
    d = sdf_smooth_union(d, g23, k=0.04)
    d = sdf_smooth_union(d, g31, k=0.04)
    d = sdf_smooth_union(d, binding, k=0.10)
    return d


def sdf_hydrogen(p, t):
    """Hydrogen atom: nucleus, electron shell, and orbital tori."""
    center = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # Nucleus
    nucleus = sdf_sphere(p, center, 0.12)

    # Electron shell (thin shell)
    shell_d = sdf_sphere(p, center, 0.8)
    shell = np.abs(shell_d) - 0.05

    # Orbital torus in XZ plane
    torus1 = sdf_torus(p, center, 0.8, 0.03)

    # Tilted second orbital torus (45 degrees around Z)
    R45 = rot_z(np.radians(45.0))
    p_r45 = apply_rotation(p, R45)
    torus2 = sdf_torus(p_r45, center, 0.8, 0.03)

    d = sdf_smooth_union(nucleus, shell, k=0.05)
    d = sdf_smooth_union(d, torus1, k=0.04)
    d = sdf_smooth_union(d, torus2, k=0.04)
    return d


def sdf_dna(p, t):
    """DNA double helix with 20 base pairs."""
    two_pi = 2.0 * np.pi
    d = np.full(p.shape[0], 1e9, dtype=np.float32)

    prev_left = None
    prev_right = None

    points_left = []
    points_right = []

    for i in range(21):
        angle = i * (two_pi / 10.0) + t * np.pi
        y_val = i * 0.12 - 1.2
        lx = 0.25 * np.cos(angle)
        lz = 0.25 * np.sin(angle)
        rx = 0.25 * np.cos(angle + np.pi)
        rz = 0.25 * np.sin(angle + np.pi)
        points_left.append(np.array([lx, y_val, lz], dtype=np.float32))
        points_right.append(np.array([rx, y_val, rz], dtype=np.float32))

    for i in range(20):
        # Left strand segment
        seg_l = sdf_capsule(p, points_left[i], points_left[i + 1], 0.04)
        d = np.minimum(d, seg_l)

        # Right strand segment
        seg_r = sdf_capsule(p, points_right[i], points_right[i + 1], 0.04)
        d = np.minimum(d, seg_r)

        # Rung connecting left to right at midpoint of each step
        mid_left = (points_left[i] + points_left[i + 1]) * 0.5
        mid_right = (points_right[i] + points_right[i + 1]) * 0.5
        rung = sdf_capsule(p, mid_left, mid_right, 0.03)
        d = np.minimum(d, rung)

    return d


def sdf_coronavirus(p, t):
    """SARS-CoV-2: central sphere + 20 spike proteins via Fibonacci sphere."""
    center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    body = sdf_sphere(p, center, 0.45)
    d = body

    # Fibonacci sphere distribution for 20 spike positions
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    n_spikes = 20
    for k in range(n_spikes):
        theta = np.arccos(1.0 - 2.0 * (k + 0.5) / n_spikes)
        phi = 2.0 * np.pi * k / golden_ratio
        dx = np.sin(theta) * np.cos(phi)
        dy = np.cos(theta)
        dz = np.sin(theta) * np.sin(phi)
        direction = np.array([dx, dy, dz], dtype=np.float32)
        spike_base = direction * 0.45
        spike_tip = direction * 0.75
        spike_stem = sdf_cylinder(p, spike_base, spike_tip, 0.04)
        spike_head = sdf_sphere(p, spike_tip, 0.07)
        d = np.minimum(d, spike_stem)
        d = np.minimum(d, spike_head)

    return d


def sdf_rbc(p, t):
    """Red blood cell: biconcave disc shape."""
    # Flatten in y direction to create disc
    p_flat = np.stack([p[:, 0], p[:, 1] * 2.5, p[:, 2]], axis=-1).astype(np.float32)
    base = sdf_torus(p_flat, np.array([0.0, 0.0, 0.0], dtype=np.float32), 0.35, 0.28)

    # Central dimple top and bottom
    dimple_top = sdf_sphere(p, np.array([0.0, 0.12, 0.0], dtype=np.float32), 0.22)
    dimple_bot = sdf_sphere(p, np.array([0.0, -0.12, 0.0], dtype=np.float32), 0.22)

    shape = sdf_subtract(base, dimple_top)
    shape = sdf_subtract(shape, dimple_bot)
    return shape


def sdf_ant(p, t):
    """Walking ant with all body parts."""
    # Body segments
    head = sdf_ellipsoid(p, np.array([-0.55, 0.05, 0.0], dtype=np.float32),
                         np.array([0.12, 0.11, 0.10], dtype=np.float32))
    thorax = sdf_ellipsoid(p, np.array([-0.1, 0.0, 0.0], dtype=np.float32),
                           np.array([0.15, 0.12, 0.13], dtype=np.float32))
    abdomen = sdf_ellipsoid(p, np.array([0.35, -0.05, 0.0], dtype=np.float32),
                            np.array([0.25, 0.18, 0.20], dtype=np.float32))

    # Connecting segments
    neck = sdf_capsule(p, np.array([-0.43, 0.0, 0.0], dtype=np.float32),
                       np.array([-0.25, 0.0, 0.0], dtype=np.float32), 0.06)
    petiole = sdf_capsule(p, np.array([0.05, -0.02, 0.0], dtype=np.float32),
                          np.array([0.15, -0.04, 0.0], dtype=np.float32), 0.05)

    # Left legs
    leg_l1 = sdf_capsule(p, np.array([-0.2, -0.05, 0.13], dtype=np.float32),
                         np.array([-0.3, -0.35, 0.25], dtype=np.float32), 0.025)
    leg_l2 = sdf_capsule(p, np.array([-0.05, -0.06, 0.13], dtype=np.float32),
                         np.array([-0.05, -0.38, 0.28], dtype=np.float32), 0.025)
    leg_l3 = sdf_capsule(p, np.array([0.1, -0.06, 0.13], dtype=np.float32),
                         np.array([0.2, -0.36, 0.25], dtype=np.float32), 0.025)

    # Right legs
    leg_r1 = sdf_capsule(p, np.array([-0.2, -0.05, -0.13], dtype=np.float32),
                         np.array([-0.3, -0.35, -0.25], dtype=np.float32), 0.025)
    leg_r2 = sdf_capsule(p, np.array([-0.05, -0.06, -0.13], dtype=np.float32),
                         np.array([-0.05, -0.38, -0.28], dtype=np.float32), 0.025)
    leg_r3 = sdf_capsule(p, np.array([0.1, -0.06, -0.13], dtype=np.float32),
                         np.array([0.2, -0.36, -0.25], dtype=np.float32), 0.025)

    # Antennae
    ant_l = sdf_capsule(p, np.array([-0.6, 0.15, 0.05], dtype=np.float32),
                        np.array([-0.9, 0.45, 0.12], dtype=np.float32), 0.018)
    ant_r = sdf_capsule(p, np.array([-0.6, 0.15, -0.05], dtype=np.float32),
                        np.array([-0.9, 0.45, -0.12], dtype=np.float32), 0.018)
    ant_l_tip = sdf_sphere(p, np.array([-0.9, 0.45, 0.12], dtype=np.float32), 0.025)
    ant_r_tip = sdf_sphere(p, np.array([-0.9, 0.45, -0.12], dtype=np.float32), 0.025)

    # Mandibles
    mand_l = sdf_capsule(p, np.array([-0.67, 0.0, 0.05], dtype=np.float32),
                         np.array([-0.80, -0.06, 0.10], dtype=np.float32), 0.020)
    mand_r = sdf_capsule(p, np.array([-0.67, 0.0, -0.05], dtype=np.float32),
                         np.array([-0.80, -0.06, -0.10], dtype=np.float32), 0.020)

    d = sdf_smooth_union(head, thorax, k=0.05)
    d = sdf_smooth_union(d, abdomen, k=0.05)
    d = sdf_smooth_union(d, neck, k=0.04)
    d = sdf_smooth_union(d, petiole, k=0.04)
    d = np.minimum(d, leg_l1)
    d = np.minimum(d, leg_l2)
    d = np.minimum(d, leg_l3)
    d = np.minimum(d, leg_r1)
    d = np.minimum(d, leg_r2)
    d = np.minimum(d, leg_r3)
    d = np.minimum(d, ant_l)
    d = np.minimum(d, ant_r)
    d = np.minimum(d, ant_l_tip)
    d = np.minimum(d, ant_r_tip)
    d = np.minimum(d, mand_l)
    d = np.minimum(d, mand_r)
    return d


def sdf_human(p, t):
    """Full standing humanoid figure."""
    # Head
    head = sdf_ellipsoid(p, np.array([0.0, 0.82, 0.0], dtype=np.float32),
                         np.array([0.11, 0.135, 0.11], dtype=np.float32))

    # Neck
    neck = sdf_capsule(p, np.array([0.0, 0.68, 0.0], dtype=np.float32),
                       np.array([0.0, 0.75, 0.0], dtype=np.float32), 0.045)

    # Torso
    torso = sdf_capsule(p, np.array([0.0, 0.25, 0.0], dtype=np.float32),
                        np.array([0.0, 0.66, 0.0], dtype=np.float32), 0.16)

    # Shoulders
    shoulder_l = sdf_sphere(p, np.array([-0.22, 0.62, 0.0], dtype=np.float32), 0.08)
    shoulder_r = sdf_sphere(p, np.array([0.22, 0.62, 0.0], dtype=np.float32), 0.08)

    # Upper arms
    uarm_l = sdf_capsule(p, np.array([-0.22, 0.62, 0.0], dtype=np.float32),
                         np.array([-0.36, 0.35, 0.04], dtype=np.float32), 0.055)
    uarm_r = sdf_capsule(p, np.array([0.22, 0.62, 0.0], dtype=np.float32),
                         np.array([0.36, 0.35, 0.04], dtype=np.float32), 0.055)

    # Forearms
    farm_l = sdf_capsule(p, np.array([-0.36, 0.35, 0.04], dtype=np.float32),
                         np.array([-0.44, 0.08, 0.05], dtype=np.float32), 0.045)
    farm_r = sdf_capsule(p, np.array([0.36, 0.35, 0.04], dtype=np.float32),
                         np.array([0.44, 0.08, 0.05], dtype=np.float32), 0.045)

    # Hands
    hand_l = sdf_ellipsoid(p, np.array([-0.46, 0.01, 0.06], dtype=np.float32),
                           np.array([0.045, 0.07, 0.03], dtype=np.float32))
    hand_r = sdf_ellipsoid(p, np.array([0.46, 0.01, 0.06], dtype=np.float32),
                           np.array([0.045, 0.07, 0.03], dtype=np.float32))

    # Hips
    hip_l = sdf_capsule(p, np.array([-0.09, 0.18, 0.0], dtype=np.float32),
                        np.array([-0.12, 0.10, 0.0], dtype=np.float32), 0.07)
    hip_r = sdf_capsule(p, np.array([0.09, 0.18, 0.0], dtype=np.float32),
                        np.array([0.12, 0.10, 0.0], dtype=np.float32), 0.07)

    # Upper legs
    uleg_l = sdf_capsule(p, np.array([-0.1, 0.1, 0.0], dtype=np.float32),
                         np.array([-0.11, -0.28, 0.02], dtype=np.float32), 0.08)
    uleg_r = sdf_capsule(p, np.array([0.1, 0.1, 0.0], dtype=np.float32),
                         np.array([0.11, -0.28, 0.02], dtype=np.float32), 0.08)

    # Lower legs
    lleg_l = sdf_capsule(p, np.array([-0.11, -0.28, 0.02], dtype=np.float32),
                         np.array([-0.10, -0.65, 0.01], dtype=np.float32), 0.065)
    lleg_r = sdf_capsule(p, np.array([0.11, -0.28, 0.02], dtype=np.float32),
                         np.array([0.10, -0.65, 0.01], dtype=np.float32), 0.065)

    # Feet
    foot_l = sdf_ellipsoid(p, np.array([-0.10, -0.70, 0.05], dtype=np.float32),
                           np.array([0.055, 0.04, 0.1], dtype=np.float32))
    foot_r = sdf_ellipsoid(p, np.array([0.10, -0.70, 0.05], dtype=np.float32),
                           np.array([0.055, 0.04, 0.1], dtype=np.float32))

    d = sdf_smooth_union(head, neck, k=0.04)
    d = sdf_smooth_union(d, torso, k=0.05)
    d = sdf_smooth_union(d, shoulder_l, k=0.04)
    d = sdf_smooth_union(d, shoulder_r, k=0.04)
    d = sdf_smooth_union(d, uarm_l, k=0.04)
    d = sdf_smooth_union(d, uarm_r, k=0.04)
    d = sdf_smooth_union(d, farm_l, k=0.03)
    d = sdf_smooth_union(d, farm_r, k=0.03)
    d = sdf_smooth_union(d, hand_l, k=0.03)
    d = sdf_smooth_union(d, hand_r, k=0.03)
    d = sdf_smooth_union(d, hip_l, k=0.05)
    d = sdf_smooth_union(d, hip_r, k=0.05)
    d = sdf_smooth_union(d, uleg_l, k=0.04)
    d = sdf_smooth_union(d, uleg_r, k=0.04)
    d = sdf_smooth_union(d, lleg_l, k=0.03)
    d = sdf_smooth_union(d, lleg_r, k=0.03)
    d = sdf_smooth_union(d, foot_l, k=0.03)
    d = sdf_smooth_union(d, foot_r, k=0.03)
    return d


def sdf_eiffel_tower(p, t):
    """Eiffel Tower: 4 legs, decks, antenna, cross-braces."""
    # Lower legs
    leg1 = sdf_capsule(p, np.array([-0.7, -1.0, -0.7], dtype=np.float32),
                       np.array([-0.05, 0.0, -0.05], dtype=np.float32), 0.06)
    leg2 = sdf_capsule(p, np.array([0.7, -1.0, -0.7], dtype=np.float32),
                       np.array([0.05, 0.0, -0.05], dtype=np.float32), 0.06)
    leg3 = sdf_capsule(p, np.array([-0.7, -1.0, 0.7], dtype=np.float32),
                       np.array([-0.05, 0.0, 0.05], dtype=np.float32), 0.06)
    leg4 = sdf_capsule(p, np.array([0.7, -1.0, 0.7], dtype=np.float32),
                       np.array([0.05, 0.0, 0.05], dtype=np.float32), 0.06)

    # Upper legs
    uleg1 = sdf_capsule(p, np.array([-0.05, 0.0, -0.05], dtype=np.float32),
                        np.array([-0.02, 0.7, -0.02], dtype=np.float32), 0.04)
    uleg2 = sdf_capsule(p, np.array([0.05, 0.0, -0.05], dtype=np.float32),
                        np.array([0.02, 0.7, -0.02], dtype=np.float32), 0.04)
    uleg3 = sdf_capsule(p, np.array([-0.05, 0.0, 0.05], dtype=np.float32),
                        np.array([-0.02, 0.7, 0.02], dtype=np.float32), 0.04)
    uleg4 = sdf_capsule(p, np.array([0.05, 0.0, 0.05], dtype=np.float32),
                        np.array([0.02, 0.7, 0.02], dtype=np.float32), 0.04)

    # Observation decks
    deck1 = sdf_box(p, np.array([0.0, -0.38, 0.0], dtype=np.float32),
                    np.array([0.55, 0.03, 0.55], dtype=np.float32))
    deck2 = sdf_box(p, np.array([0.0, 0.12, 0.0], dtype=np.float32),
                    np.array([0.25, 0.03, 0.25], dtype=np.float32))
    deck3 = sdf_box(p, np.array([0.0, 0.55, 0.0], dtype=np.float32),
                    np.array([0.10, 0.025, 0.10], dtype=np.float32))

    # Antenna
    antenna = sdf_capsule(p, np.array([0.0, 0.7, 0.0], dtype=np.float32),
                          np.array([0.0, 1.0, 0.0], dtype=np.float32), 0.015)

    # Cross-braces
    brace_x1 = sdf_box(p, np.array([0.0, -0.7, 0.0], dtype=np.float32),
                       np.array([0.58, 0.015, 0.02], dtype=np.float32))
    brace_z1 = sdf_box(p, np.array([0.0, -0.7, 0.0], dtype=np.float32),
                       np.array([0.02, 0.015, 0.58], dtype=np.float32))
    brace_x2 = sdf_box(p, np.array([0.0, -0.1, 0.0], dtype=np.float32),
                       np.array([0.28, 0.012, 0.02], dtype=np.float32))
    brace_z2 = sdf_box(p, np.array([0.0, -0.1, 0.0], dtype=np.float32),
                       np.array([0.02, 0.012, 0.28], dtype=np.float32))

    d = np.minimum(leg1, leg2)
    d = np.minimum(d, leg3)
    d = np.minimum(d, leg4)
    d = np.minimum(d, uleg1)
    d = np.minimum(d, uleg2)
    d = np.minimum(d, uleg3)
    d = np.minimum(d, uleg4)
    d = np.minimum(d, deck1)
    d = np.minimum(d, deck2)
    d = np.minimum(d, deck3)
    d = np.minimum(d, antenna)
    d = np.minimum(d, brace_x1)
    d = np.minimum(d, brace_z1)
    d = np.minimum(d, brace_x2)
    d = np.minimum(d, brace_z2)
    return d


def sdf_everest(p, t):
    """Mount Everest with procedural rocky surface."""
    # Main peak cone
    main_cone = sdf_cone(p, np.array([0.0, 1.0, 0.0], dtype=np.float32),
                         np.array([0.0, -0.8, 0.0], dtype=np.float32), 1.2)

    # Noise displacement for rocky appearance
    noise_offset = 0.04 * np.sin(p[:, 0] * 8.0) * np.cos(p[:, 2] * 8.0)
    main_cone_displaced = main_cone + noise_offset

    # Secondary ridges
    ridge1 = sdf_cone(p, np.array([0.3, 0.7, 0.1], dtype=np.float32),
                      np.array([0.5, -0.5, 0.2], dtype=np.float32), 0.5)
    ridge2 = sdf_cone(p, np.array([-0.4, 0.5, -0.1], dtype=np.float32),
                      np.array([-0.6, -0.6, -0.2], dtype=np.float32), 0.45)

    # Rocky outcrops
    outcrop1 = sdf_box(p, np.array([0.2, 0.3, 0.3], dtype=np.float32),
                       np.array([0.08, 0.15, 0.08], dtype=np.float32))
    outcrop2 = sdf_box(p, np.array([-0.15, 0.4, -0.25], dtype=np.float32),
                       np.array([0.07, 0.12, 0.07], dtype=np.float32))

    # Snow cap
    snow = sdf_sphere(p, np.array([0.0, 0.88, 0.0], dtype=np.float32), 0.20)

    d = sdf_union(main_cone_displaced, ridge1)
    d = sdf_union(d, ridge2)
    d = sdf_union(d, outcrop1)
    d = sdf_union(d, outcrop2)
    d = sdf_smooth_union(d, snow, k=0.06)
    return d


def sdf_earth(p, t):
    """Earth with oblate shape, slow rotation, and atmosphere."""
    # Slow rotation around Y
    R = rot_y(t * 2.0 * np.pi * 0.1)
    p_rot = apply_rotation(p, R)

    # Oblate spheroid
    main = sdf_ellipsoid(p_rot, np.array([0.0, 0.0, 0.0], dtype=np.float32),
                         np.array([1.005, 1.0, 1.005], dtype=np.float32))

    # Thin atmosphere shell
    atmo_d = sdf_sphere(p, np.array([0.0, 0.0, 0.0], dtype=np.float32), 1.06)
    atmosphere = np.abs(atmo_d) - 0.015

    d = sdf_smooth_union(main, atmosphere, k=0.05)
    return d


def sdf_jupiter(p, t):
    """Jupiter with oblate shape, band rotation, and Great Red Spot."""
    # Slow rotation around Y
    R = rot_y(t * 0.5)
    p_rot = apply_rotation(p, R)

    # Oblate spheroid
    oblate = sdf_ellipsoid(p_rot, np.array([0.0, 0.0, 0.0], dtype=np.float32),
                           np.array([1.065, 1.0, 1.065], dtype=np.float32))

    # Great Red Spot
    spot_angle = t * 0.3
    spot_pos = np.array([np.cos(spot_angle) * 1.07, -0.22,
                         np.sin(spot_angle) * 1.07], dtype=np.float32)
    spot = sdf_sphere(p, spot_pos, 0.12)

    d = sdf_smooth_union(oblate, spot, k=0.05)
    return d


def sdf_sun(p, t):
    """The Sun with corona and solar prominences."""
    center = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # Main sphere
    main = sdf_sphere(p, center, 1.0)

    # Corona shell
    corona_d = sdf_sphere(p, center, 1.15)
    corona = np.abs(corona_d) - 0.08

    # Solar prominences (tilted tori)
    R30 = rot_x(np.radians(30.0))
    p_r30 = apply_rotation(p, R30)
    prom1 = sdf_torus(p_r30, center, 1.05, 0.04)

    R70 = rot_x(np.radians(70.0))
    p_r70 = apply_rotation(p, R70)
    prom2 = sdf_torus(p_r70, center, 1.08, 0.035)

    R150 = rot_x(np.radians(150.0))
    p_r150 = apply_rotation(p, R150)
    prom3 = sdf_torus(p_r150, center, 1.02, 0.05)

    d = sdf_smooth_union(main, corona, k=0.06)
    d = sdf_smooth_union(d, prom1, k=0.04)
    d = sdf_smooth_union(d, prom2, k=0.04)
    d = sdf_smooth_union(d, prom3, k=0.04)
    return d


def sdf_solar_system(p, t):
    """Solar system: sun, 8 planet orbits, 8 planets."""
    center = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # Sun
    sun = sdf_sphere(p, center, 0.08)
    d = sun

    # Orbital radii
    orbit_radii = [0.15, 0.22, 0.30, 0.40, 0.65, 0.82, 0.94, 1.05]
    # Relative orbital speeds (normalised)
    speeds = [4.74, 3.50, 2.98, 2.41, 1.31, 0.97, 0.68, 0.54]
    planet_sizes = [0.008, 0.014, 0.015, 0.010, 0.040, 0.032, 0.025, 0.024]

    for i, (r_orb, spd, p_sz) in enumerate(zip(orbit_radii, speeds, planet_sizes)):
        # Thin orbit torus in XZ plane
        orbit_torus = sdf_torus(p, center, r_orb, 0.008)
        d = np.minimum(d, orbit_torus)

        # Planet position
        angle = t * 2.0 * np.pi * spd
        px = r_orb * np.cos(angle)
        pz = r_orb * np.sin(angle)
        planet_pos = np.array([px, 0.0, pz], dtype=np.float32)
        planet = sdf_sphere(p, planet_pos, p_sz)
        d = np.minimum(d, planet)

    return d


def sdf_lightyear(p, t):
    """Light year: measurement tube with markers, beams, and stars."""
    # Main central tube
    main_tube = sdf_cylinder(p,
                             np.array([0.0, -1.0, 0.0], dtype=np.float32),
                             np.array([0.0, 1.0, 0.0], dtype=np.float32), 0.06)
    d = main_tube

    # Distance marker rings
    for i in range(12):
        y_pos = -0.9 + i * 0.15
        marker = sdf_box(p, np.array([0.0, y_pos, 0.0], dtype=np.float32),
                         np.array([0.12, 0.008, 0.008], dtype=np.float32))
        d = np.minimum(d, marker)

    # Parallel light beam lines
    beam_offsets = [(-0.15, 0.0), (0.15, 0.0), (0.0, -0.15), (0.0, 0.15)]
    for ox, oz in beam_offsets:
        beam = sdf_cylinder(p,
                            np.array([ox, -1.0, oz], dtype=np.float32),
                            np.array([ox, 1.0, oz], dtype=np.float32), 0.015)
        d = np.minimum(d, beam)

    # Decorative stars
    star_positions = [
        (0.4, -0.8, 0.2), (-0.3, 0.5, 0.35), (0.5, 0.1, -0.4),
        (-0.5, -0.3, -0.2), (0.3, 0.7, -0.3), (-0.4, -0.6, 0.3),
        (0.45, -0.2, 0.4), (-0.35, 0.3, -0.45),
    ]
    for sx, sy, sz in star_positions:
        star = sdf_sphere(p, np.array([sx, sy, sz], dtype=np.float32), 0.02)
        d = np.minimum(d, star)

    return d


def sdf_milkyway(p, t):
    """Milky Way galaxy with disc, central bulge, and 4 spiral arms."""
    center = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # Main disc
    disc = sdf_ellipsoid(p, center, np.array([1.0, 0.08, 1.0], dtype=np.float32))

    # Central bulge
    bulge = sdf_sphere(p, center, 0.22)

    d = sdf_smooth_union(disc, bulge, k=0.08)

    # 4 spiral arms (6 segments each)
    for k_arm in range(4):
        start_angle = k_arm * (np.pi / 2.0)
        for j in range(6):
            angle0 = start_angle + j * 0.4
            angle1 = start_angle + (j + 1) * 0.4
            r0 = 0.15 + j * 0.12
            r1 = 0.15 + (j + 1) * 0.12
            y0 = (j - 3) * 0.01
            y1 = (j + 1 - 3) * 0.01
            a0 = np.array([r0 * np.cos(angle0), y0, r0 * np.sin(angle0)], dtype=np.float32)
            a1 = np.array([r1 * np.cos(angle1), y1, r1 * np.sin(angle1)], dtype=np.float32)
            arm_seg = sdf_capsule(p, a0, a1, 0.02)
            d = sdf_smooth_union(d, arm_seg, k=0.04)

    return d


def sdf_local_group(p, t):
    """Local Group: Milky Way, Andromeda, and dwarf galaxies."""
    center = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # Milky Way disc
    milkyway = sdf_ellipsoid(p, center, np.array([0.35, 0.04, 0.35], dtype=np.float32))

    # Andromeda — rotated 30 degrees around Y
    R_and = rot_y(np.pi / 6.0)
    p_and = apply_rotation(p - np.array([0.8, 0.1, 0.3], dtype=np.float32), R_and)
    andromeda = sdf_ellipsoid(p_and, center, np.array([0.30, 0.035, 0.30], dtype=np.float32))

    d = np.minimum(milkyway, andromeda)

    # Dwarf galaxies
    dwarf_positions_large = [
        (0.4, 0.1, 0.5), (-0.5, -0.1, 0.2), (0.2, 0.2, -0.7),
        (-0.6, 0.15, -0.4), (0.7, -0.15, 0.5), (-0.3, -0.2, -0.6),
        (0.5, 0.1, -0.5), (-0.7, 0.05, 0.3),
    ]
    for dx, dy, dz in dwarf_positions_large:
        dwarf = sdf_sphere(p, np.array([dx, dy, dz], dtype=np.float32), 0.04)
        d = np.minimum(d, dwarf)

    dwarf_positions_small = [
        (-0.2, 0.3, 0.4), (0.4, -0.2, -0.3),
        (-0.45, 0.2, 0.55), (0.6, -0.1, -0.25),
    ]
    for dx, dy, dz in dwarf_positions_small:
        dwarf = sdf_sphere(p, np.array([dx, dy, dz], dtype=np.float32), 0.03)
        d = np.minimum(d, dwarf)

    return d


def sdf_observable_universe(p, t):
    """Observable Universe: boundary shell, cosmic filaments, galaxy clusters."""
    # Slow rotation
    R = rot_y(t * 0.2 * np.pi)
    p_rot = apply_rotation(p, R)

    # Outer boundary thin shell
    boundary_d = sdf_sphere(p_rot, np.array([0.0, 0.0, 0.0], dtype=np.float32), 1.0)
    boundary = np.abs(boundary_d) - 0.02

    d = boundary

    # Cosmic filaments
    filament_endpoints = [
        (np.array([-0.8, 0.2, 0.3], dtype=np.float32),   np.array([0.7, -0.1, -0.4], dtype=np.float32)),
        (np.array([-0.5, -0.6, 0.3], dtype=np.float32),  np.array([0.5, 0.7, -0.2], dtype=np.float32)),
        (np.array([0.2, -0.7, 0.5], dtype=np.float32),   np.array([-0.1, 0.6, -0.6], dtype=np.float32)),
        (np.array([0.6, 0.3, -0.7], dtype=np.float32),   np.array([-0.7, -0.2, 0.4], dtype=np.float32)),
        (np.array([-0.4, 0.5, -0.6], dtype=np.float32),  np.array([0.3, -0.6, 0.5], dtype=np.float32)),
        (np.array([0.7, -0.5, 0.2], dtype=np.float32),   np.array([-0.6, 0.3, -0.3], dtype=np.float32)),
        (np.array([0.5, 0.6, 0.4], dtype=np.float32),    np.array([-0.4, -0.5, -0.5], dtype=np.float32)),
        (np.array([-0.3, 0.4, 0.7], dtype=np.float32),   np.array([0.3, -0.4, -0.7], dtype=np.float32)),
    ]

    for a_end, b_end in filament_endpoints:
        filament = sdf_capsule(p_rot, a_end, b_end, 0.03)
        d = np.minimum(d, filament)

    # Galaxy cluster nodes at filament midpoints
    for a_end, b_end in filament_endpoints:
        mid = (a_end + b_end) * 0.5
        cluster = sdf_sphere(p_rot, mid, 0.05)
        d = np.minimum(d, cluster)

    # Extra cluster nodes
    extra_clusters = [
        np.array([0.3, 0.3, 0.3], dtype=np.float32),
        np.array([-0.3, -0.3, -0.3], dtype=np.float32),
        np.array([0.4, -0.3, 0.2], dtype=np.float32),
        np.array([-0.2, 0.4, -0.3], dtype=np.float32),
    ]
    for ec in extra_clusters:
        cluster = sdf_sphere(p_rot, ec, 0.05)
        d = np.minimum(d, cluster)

    return d

# ---------------------------------------------------------------------------
# Section 8: RAYMARCHING ENGINE
# ---------------------------------------------------------------------------


def raymarch(ray_origins, ray_dirs, sdf_func, t_anim):
    """
    Vectorized raymarcher.
    ray_origins: [H, W, 3]
    ray_dirs: [H, W, 3]
    Returns: hit_dist [H,W], hit_mask [H,W]
    """
    H, W = ray_origins.shape[:2]
    t = np.full((H, W), 0.1, dtype=np.float32)
    hit_mask = np.zeros((H, W), dtype=bool)

    for step in range(MAX_STEPS):
        active = (t < MAX_DIST) & (~hit_mask)
        if not np.any(active):
            break
        p = ray_origins + ray_dirs * t[..., np.newaxis]
        p_flat = p[active].reshape(-1, 3).astype(np.float32)
        d = sdf_func(p_flat, t_anim).astype(np.float32)
        step_mask = np.zeros((H, W), dtype=np.float32)
        step_mask[active] = d
        new_hits = active & (step_mask < SURF_DIST) & (step_mask > -SURF_DIST * 5)
        hit_mask |= new_hits
        t = np.where(active & ~new_hits,
                     t + np.maximum(step_mask, SURF_DIST * 0.3), t)

    t = np.where(hit_mask, t, MAX_DIST)
    return t, hit_mask


def estimate_normal(p_flat, sdf_func, t_anim, eps=0.003):
    """Central differences normal estimation for flat array of hit points [N,3]."""
    eps_arr = np.float32(eps)
    dx = np.array([eps_arr, 0, 0], dtype=np.float32)
    dy = np.array([0, eps_arr, 0], dtype=np.float32)
    dz = np.array([0, 0, eps_arr], dtype=np.float32)
    nx = sdf_func(p_flat + dx, t_anim) - sdf_func(p_flat - dx, t_anim)
    ny = sdf_func(p_flat + dy, t_anim) - sdf_func(p_flat - dy, t_anim)
    nz = sdf_func(p_flat + dz, t_anim) - sdf_func(p_flat - dz, t_anim)
    n = np.stack([nx, ny, nz], axis=-1)
    nl = np.sqrt(np.sum(n ** 2, axis=-1, keepdims=True)) + 1e-8
    return (n / nl).astype(np.float32)

# ---------------------------------------------------------------------------
# Section 9: MATERIAL & LIGHTING SYSTEM
# ---------------------------------------------------------------------------

def phong_lighting(normals, view_dir, light_dir, ambient=0.15, diffuse=0.7,
                   specular=0.4, shininess=32.0, rim_strength=0.3):
    """Full Phong shading with rim lighting. All arrays [N,3] or scalar. Returns [N]."""
    ndotl = np.maximum(np.sum(normals * light_dir, axis=-1), 0.0)
    light = ambient + diffuse * ndotl

    half_vec = light_dir + view_dir
    half_len = np.sqrt(np.sum(half_vec ** 2, axis=-1, keepdims=True)) + 1e-8
    half_vec = half_vec / half_len
    ndoth = np.maximum(np.sum(normals * half_vec, axis=-1), 0.0)
    light += specular * (ndoth ** shininess)

    rim = 1.0 - np.maximum(np.sum(normals * view_dir, axis=-1), 0.0)
    light += rim_strength * (rim ** 3)
    return light


def compute_ao(hit_pos, normals, sdf_func, t_anim):
    """Ambient occlusion via cone sampling. hit_pos/normals are [N,3]."""
    ao = np.zeros(hit_pos.shape[0], dtype=np.float32)
    for i in range(1, 6):
        step = 0.06 * i
        p = hit_pos + normals * step
        d = sdf_func(p, t_anim).astype(np.float32)
        ao += (step - np.clip(d, 0.0, step)) / (2.0 ** i)
    return np.clip(1.0 - 2.5 * ao, 0.0, 1.0)

# ---------------------------------------------------------------------------
# Section 10: RENDER PIPELINE
# ---------------------------------------------------------------------------

def render_frame(sdf_func, obj_data, camera_pos, camera_target, t_anim, scale=1.0):
    """
    Render a single frame.
    Returns np.array [HEIGHT, WIDTH, 3] uint8.
    """
    camera_pos = np.asarray(camera_pos, dtype=np.float32)
    camera_target = np.asarray(camera_target, dtype=np.float32)

    fov = np.pi / 3.8
    aspect = WIDTH / HEIGHT

    forward = camera_target - camera_pos
    fwd_len = np.linalg.norm(forward)
    if fwd_len < 1e-8:
        forward = np.array([0, 0, -1], dtype=np.float32)
    else:
        forward = forward / fwd_len

    world_up = np.array([0, 1, 0], dtype=np.float32)
    if abs(np.dot(forward, world_up)) > 0.99:
        world_up = np.array([0, 0, 1], dtype=np.float32)

    right = np.cross(forward, world_up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)
    up = up / (np.linalg.norm(up) + 1e-8)

    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    ndc_x = (2.0 * (xs + 0.5) / WIDTH - 1.0) * np.tan(fov / 2) * aspect
    ndc_y = -(2.0 * (ys + 0.5) / HEIGHT - 1.0) * np.tan(fov / 2)

    ray_dirs = (forward[np.newaxis, np.newaxis, :]
                + right[np.newaxis, np.newaxis, :] * ndc_x[:, :, np.newaxis]
                + up[np.newaxis, np.newaxis, :] * ndc_y[:, :, np.newaxis])
    ray_lens = np.sqrt(np.sum(ray_dirs ** 2, axis=-1, keepdims=True)) + 1e-8
    ray_dirs = (ray_dirs / ray_lens).astype(np.float32)
    ray_origins = np.broadcast_to(camera_pos, (HEIGHT, WIDTH, 3)).copy().astype(np.float32)

    def scaled_sdf(p_in, ta):
        return sdf_func(p_in / scale, ta) * scale

    hit_dist, hit_mask = raymarch(ray_origins, ray_dirs, scaled_sdf, t_anim)

    # Build color image
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)

    if np.any(hit_mask):
        hit_pos_flat = (ray_origins[hit_mask]
                        + ray_dirs[hit_mask] * hit_dist[hit_mask, np.newaxis]).astype(np.float32)

        normals_flat = estimate_normal(hit_pos_flat, scaled_sdf, t_anim)

        view_dir_flat = (-ray_dirs[hit_mask]).astype(np.float32)

        light_pos = np.array([3.0, 4.0, 5.0], dtype=np.float32)
        light_dir_flat = light_pos - hit_pos_flat
        ld_len = np.sqrt(np.sum(light_dir_flat ** 2, axis=-1, keepdims=True)) + 1e-8
        light_dir_flat = light_dir_flat / ld_len

        lighting = phong_lighting(normals_flat, view_dir_flat, light_dir_flat,
                                  ambient=0.18, diffuse=0.65, specular=0.45,
                                  shininess=40.0, rim_strength=0.35)

        ao = compute_ao(hit_pos_flat, normals_flat, scaled_sdf, t_anim)
        lighting = lighting * ao

        base_color = np.array(obj_data['color'], dtype=np.float32) / 255.0
        lit_color = base_color[np.newaxis, :] * lighting[:, np.newaxis]

        # Second fill light (blue-ish from opposite side)
        fill_light_dir = np.array([-0.5, 0.3, -0.7], dtype=np.float32)
        fill_light_dir = fill_light_dir / (np.linalg.norm(fill_light_dir) + 1e-8)
        fill_ndotl = np.maximum(np.sum(normals_flat * fill_light_dir, axis=-1), 0.0)
        fill_color = np.array([0.1, 0.15, 0.3], dtype=np.float32)
        lit_color += fill_color[np.newaxis, :] * (fill_ndotl * 0.25)[:, np.newaxis]

        lit_color = np.clip(lit_color, 0.0, 1.0)
        frame[hit_mask] = lit_color

    # Glow/bloom
    frame_uint8 = np.clip(frame * 255, 0, 255).astype(np.uint8)
    frame_float = frame_uint8.astype(np.float32)

    if np.any(hit_mask):
        glow_color = np.array(obj_data['glow_color'], dtype=np.float32) / 255.0
        mask_float = hit_mask.astype(np.float32)
        glow_mask = gaussian_filter(mask_float, sigma=12.0)
        for c in range(3):
            frame_float[:, :, c] += glow_mask * glow_color[c] * 80.0

    frame_float = np.clip(frame_float, 0, 255).astype(np.uint8)
    return frame_float

# ---------------------------------------------------------------------------
# Section 11: STARFIELD BACKGROUND
# ---------------------------------------------------------------------------

def generate_starfield():
    """Generate a static starfield + nebula background."""
    bg = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)

    rng = np.random.RandomState(42)
    n_stars = 3000
    star_x = rng.randint(0, WIDTH, n_stars)
    star_y = rng.randint(0, HEIGHT, n_stars)
    star_brightness = np.clip(rng.exponential(0.5, n_stars), 0.0, 1.0)

    for i in range(n_stars):
        b = float(star_brightness[i])
        x, y = int(star_x[i]), int(star_y[i])
        bg[y, x] = [b, b * 0.9, b * 0.8]
        if b > 0.7 and x > 1 and y > 1 and x < WIDTH - 2 and y < HEIGHT - 2:
            bg[y - 1, x] = [b * 0.4, b * 0.36, b * 0.32]
            bg[y + 1, x] = [b * 0.4, b * 0.36, b * 0.32]
            bg[y, x - 1] = [b * 0.4, b * 0.36, b * 0.32]
            bg[y, x + 1] = [b * 0.4, b * 0.36, b * 0.32]

    nebula_colors = [
        np.array([0.4, 0.1, 0.6]),
        np.array([0.1, 0.2, 0.7]),
        np.array([0.6, 0.1, 0.2]),
        np.array([0.1, 0.5, 0.4]),
        np.array([0.3, 0.2, 0.5]),
    ]
    for nc in nebula_colors:
        cx = rng.randint(100, WIDTH - 100)
        cy = rng.randint(100, HEIGHT - 100)
        sigma = rng.randint(120, 320)
        ys, xs = np.ogrid[:HEIGHT, :WIDTH]
        nebula = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
        bg += nebula[:, :, np.newaxis] * nc[np.newaxis, np.newaxis, :] * 0.18

    for c in range(3):
        bg[:, :, c] = gaussian_filter(bg[:, :, c], sigma=1.2)

    return np.clip(bg * 255, 0, 255).astype(np.uint8)


def blend_with_starfield(frame, starfield, alpha=1.0):
    """Composite rendered frame over starfield background."""
    frame_f = frame.astype(np.float32)
    star_f = starfield.astype(np.float32)

    luminance = np.sum(frame_f, axis=-1)
    obj_mask = (luminance > 8.0).astype(np.float32)

    obj_mask_smooth = gaussian_filter(obj_mask, sigma=2.0)
    obj_mask_smooth = np.clip(obj_mask_smooth, 0, 1)[:, :, np.newaxis]

    result = (frame_f * obj_mask_smooth
              + star_f * (1.0 - obj_mask_smooth))
    result = (result * alpha
              + star_f * (1.0 - alpha) * 0.3
              + np.array(BG_COLOR, dtype=np.float32)[np.newaxis, np.newaxis, :] * (1.0 - alpha) * 0.7)
    return np.clip(result, 0, 255).astype(np.uint8)

# ---------------------------------------------------------------------------
# Section 12: FONT LOADING & TEXT RENDERING
# ---------------------------------------------------------------------------

def load_fonts():
    """Load fonts, falling back to default if not found."""
    fonts = {}
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    found_path = None
    for path in font_paths:
        if os.path.exists(path):
            found_path = path
            break

    sizes = {'title': 44, 'name': 66, 'subtitle': 34, 'body': 27, 'small': 21}
    for key, size in sizes.items():
        if found_path:
            try:
                fonts[key] = ImageFont.truetype(found_path, size)
                continue
            except Exception:
                pass
        fonts[key] = ImageFont.load_default()
    return fonts


def draw_text_with_glow(draw, text, position, font, color, glow_color, glow_radius=4):
    """Draw text with a soft glow effect using multiple offset copies."""
    x, y = position
    for dx in range(-glow_radius, glow_radius + 1):
        for dy in range(-glow_radius, glow_radius + 1):
            dist_sq = dx * dx + dy * dy
            if dist_sq <= glow_radius * glow_radius and dist_sq > 0:
                alpha = 1.0 - (dist_sq ** 0.5) / glow_radius
                gc = tuple(int(c * alpha * 0.55) for c in glow_color) + (255,)
                draw.text((x + dx, y + dy), text, font=font, fill=gc[:3])
    draw.text((x, y), text, font=font, fill=color)

# ---------------------------------------------------------------------------
# Section 13: SCALE BAR & UI ELEMENTS
# ---------------------------------------------------------------------------

def draw_scale_bar(draw, fonts, current_idx, n_objects):
    """Draw logarithmic scale progress bar at bottom."""
    bar_x, bar_y = 60, HEIGHT - 55
    bar_w = WIDTH - 120
    bar_h = 7

    draw.rectangle([bar_x - 1, bar_y - 1, bar_x + bar_w + 1, bar_y + bar_h + 1],
                   fill=(20, 20, 40))
    draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], fill=(40, 40, 70))

    progress = current_idx / max(n_objects - 1, 1)
    fill_w = int(bar_w * progress)
    if fill_w > 0:
        for px in range(fill_w):
            t_bar = px / max(fill_w, 1)
            r = int(60 + t_bar * 80)
            g = int(100 + t_bar * 80)
            b = int(220 + t_bar * 35)
            draw.line([(bar_x + px, bar_y), (bar_x + px, bar_y + bar_h)],
                      fill=(r, g, min(b, 255)))

    for i in range(n_objects):
        tx = bar_x + int(bar_w * i / max(n_objects - 1, 1))
        col = (180, 180, 255) if i <= current_idx else (80, 80, 120)
        draw.rectangle([tx - 1, bar_y - 4, tx + 1, bar_y + bar_h + 4], fill=col)


def draw_ui_overlay(img, draw, fonts, obj_data, obj_idx, n_objects, phase, phase_t):
    """Draw all UI overlay elements onto the frame."""
    # Title bar with semi-transparent bg
    draw.rectangle([0, 0, WIDTH, 75], fill=(4, 3, 18, 180))

    title = "SIZE COMPARISON: FROM QUARK TO OBSERVABLE UNIVERSE"
    draw_text_with_glow(draw, title, (WIDTH // 2 - 470, 15),
                        fonts['title'], (240, 240, 255), (80, 80, 200), glow_radius=3)

    counter = f"#{obj_idx + 1:02d} / {n_objects}"
    draw_text_with_glow(draw, counter, (WIDTH - 170, 22),
                        fonts['subtitle'], (180, 180, 220), (60, 60, 120))

    # Left panel — fade based on phase
    alpha = 1.0
    if phase == 'appear':
        alpha = float(phase_t)
    elif phase == 'transition':
        alpha = max(0.0, 1.0 - float(phase_t) * 2.0)

    if alpha > 0.02:
        panel_x = 55
        color = obj_data['color']
        glow = obj_data['glow_color']

        # Category
        cat_color = tuple(min(255, int(c * 0.75)) for c in color)
        draw_text_with_glow(draw, obj_data['category'],
                            (panel_x, HEIGHT // 2 - 195),
                            fonts['subtitle'], cat_color, glow, glow_radius=2)

        # Object name
        draw_text_with_glow(draw, obj_data['name'],
                            (panel_x, HEIGHT // 2 - 155),
                            fonts['name'], color, glow, glow_radius=7)

        # SI and US labels
        draw_text_with_glow(draw, f"SI: {obj_data['si_label']}",
                            (panel_x, HEIGHT // 2 - 65),
                            fonts['body'], (210, 220, 255), (50, 60, 130))
        draw_text_with_glow(draw, f"US: {obj_data['us_label']}",
                            (panel_x, HEIGHT // 2 - 28),
                            fonts['body'], (220, 210, 200), (70, 60, 50))

        # Divider
        draw.line([(panel_x, HEIGHT // 2 + 8), (panel_x + 420, HEIGHT // 2 + 8)],
                  fill=(80, 80, 120), width=1)

        # Facts
        for i, fact in enumerate(obj_data['facts'][:3]):
            draw_text_with_glow(draw, f"• {fact}",
                                (panel_x, HEIGHT // 2 + 22 + i * 38),
                                fonts['body'], (195, 200, 210), (35, 40, 75))

    draw_scale_bar(draw, fonts, obj_idx, n_objects)
    draw.text((WIDTH - 240, HEIGHT - 28), "Generated with Python + NumPy",
              font=fonts['small'], fill=(60, 60, 90))

# ---------------------------------------------------------------------------
# Section 14: CAMERA SYSTEM
# ---------------------------------------------------------------------------

def get_camera_pos(obj_data, phase, phase_t, prev_obj_data=None):
    """Compute camera position based on current animation phase."""
    base_dist = 3.2

    if phase == 'appear':
        t_ease = ease_in_out_cubic(float(phase_t))
        dist = base_dist * (1.0 + (1.0 - t_ease) * 2.5)
        angle = float(phase_t) * 0.3
        return np.array([np.sin(angle) * dist * 0.15, 0.4, dist], dtype=np.float32)

    elif phase == 'hold':
        angle = float(phase_t) * 0.4
        return np.array([np.sin(angle) * 0.5 * base_dist, 0.3,
                         np.cos(angle) * base_dist], dtype=np.float32)

    elif phase == 'transition':
        if phase_t < 0.5:
            t_ease = ease_in_out_cubic(float(phase_t) * 2.0)
            dist = base_dist * (1.0 + t_ease * 9.0)
            return np.array([0.0, 0.5, dist], dtype=np.float32)
        else:
            t_ease = ease_in_out_cubic((float(phase_t) - 0.5) * 2.0)
            dist = base_dist * (1.0 + (1.0 - t_ease) * 9.0)
            return np.array([0.0, 0.4, dist], dtype=np.float32)

    return np.array([0.0, 0.4, base_dist], dtype=np.float32)

# ---------------------------------------------------------------------------
# Section 15: FRAME GENERATOR
# ---------------------------------------------------------------------------

def get_sdf_funcs():
    return [
        sdf_quark, sdf_proton, sdf_hydrogen, sdf_dna, sdf_coronavirus,
        sdf_rbc, sdf_ant, sdf_human, sdf_eiffel_tower, sdf_everest,
        sdf_earth, sdf_jupiter, sdf_sun, sdf_solar_system, sdf_lightyear,
        sdf_milkyway, sdf_local_group, sdf_observable_universe,
    ]


def generate_object_frames(obj_idx, obj_data, starfield, fonts):
    """Generate all frames for one object (appear + hold + transition phases)."""
    sdf_funcs = get_sdf_funcs()
    sdf_func = sdf_funcs[obj_idx]
    n_objects = len(sdf_funcs)
    frames = []

    # Appear phase
    for f in range(F_APPEAR):
        t = f / max(F_APPEAR - 1, 1)
        scale = 0.3 + 0.7 * ease_out_bounce(t)
        camera_pos = get_camera_pos(obj_data, 'appear', t)
        frame = render_frame(sdf_func, obj_data, camera_pos,
                             np.array([0, 0, 0], dtype=np.float32),
                             t_anim=t, scale=scale)
        composite = blend_with_starfield(frame, starfield, alpha=min(t * 2, 1.0))
        img = Image.fromarray(composite)
        draw = ImageDraw.Draw(img)
        draw_ui_overlay(img, draw, fonts, obj_data, obj_idx, n_objects, 'appear', t)
        frames.append(np.array(img))

    # Hold phase
    for f in range(F_HOLD):
        t = f / max(F_HOLD - 1, 1)
        camera_pos = get_camera_pos(obj_data, 'hold', t)
        frame = render_frame(sdf_func, obj_data, camera_pos,
                             np.array([0, 0, 0], dtype=np.float32),
                             t_anim=t, scale=1.0)
        composite = blend_with_starfield(frame, starfield, alpha=1.0)
        img = Image.fromarray(composite)
        draw = ImageDraw.Draw(img)
        draw_ui_overlay(img, draw, fonts, obj_data, obj_idx, n_objects, 'hold', t)
        frames.append(np.array(img))

    # Transition phase
    for f in range(F_TRANSITION):
        t = f / max(F_TRANSITION - 1, 1)
        camera_pos = get_camera_pos(obj_data, 'transition', t)
        if t < 0.5:
            frame = render_frame(sdf_func, obj_data, camera_pos,
                                 np.array([0, 0, 0], dtype=np.float32),
                                 t_anim=t, scale=1.0)
            bg_alpha = 1.0 - t * 1.4
        else:
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            bg_alpha = 0.15 + (t - 0.5) * 1.7
        composite = blend_with_starfield(frame, starfield, alpha=max(0.05, bg_alpha))
        img = Image.fromarray(composite)
        draw = ImageDraw.Draw(img)
        draw_ui_overlay(img, draw, fonts, obj_data, obj_idx, n_objects, 'transition', t)
        frames.append(np.array(img))

    return frames

# ---------------------------------------------------------------------------
# Section 16: MAIN LOOP + CLEANUP
# ---------------------------------------------------------------------------

def print_summary(output_file, n_frames):
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output file : {output_file}")
    if os.path.exists(output_file):
        size_mb = os.path.getsize(output_file) / 1024 / 1024
        print(f"File size   : {size_mb:.1f} MB")
    duration = n_frames * _FRAME_REPEAT / FPS_OUTPUT  # each rendered frame repeated _FRAME_REPEAT times
    total_output_frames = n_frames * _FRAME_REPEAT
    print(f"Frames      : {n_frames} rendered → {total_output_frames} output at {FPS_OUTPUT} fps")
    print(f"Duration    : {duration:.1f}s  ({duration / 60:.1f} min)")
    print(f"Objects     : {len(OBJECTS)}")
    print("=" * 60)


def main():
    print("=" * 60)
    print("  SIZE COMPARISON: FROM QUARK TO OBSERVABLE UNIVERSE")
    print("=" * 60)
    print(f"Resolution : {WIDTH}x{HEIGHT} @ {FPS_RENDER} fps render, {FPS_OUTPUT} fps output")
    print(f"Objects    : {len(OBJECTS)}")
    print(f"Est. frames: {len(OBJECTS) * F_PER_OBJ} rendered")
    print()

    print("[1/3] Generating starfield background...")
    starfield = generate_starfield()

    print("[2/3] Loading fonts...")
    fonts = load_fonts()

    print(f"[3/3] Rendering {len(OBJECTS)} objects...")
    all_frames = []

    for i, obj in enumerate(tqdm(OBJECTS, desc="Objects", unit="obj")):
        tqdm.write(f"  -> [{i + 1:02d}/{len(OBJECTS)}] {obj['name']}")
        frames = generate_object_frames(i, obj, starfield, fonts)
        all_frames.extend(frames)
        gc.collect()

    output_file = "size_comparison_3d.mp4"
    print(f"\nCompiling {len(all_frames)} frames → {output_file} ...")

    writer = imageio.get_writer(
        output_file,
        fps=FPS_OUTPUT,
        codec='libx264',
        quality=8,
        macro_block_size=1,
    )
    for frame in tqdm(all_frames, desc="Writing", unit="frame"):
        for _ in range(_FRAME_REPEAT):  # repeat each rendered frame to reach FPS_OUTPUT
            writer.append_data(frame)
    writer.close()

    print_summary(output_file, len(all_frames))


if __name__ == "__main__":
    main()
