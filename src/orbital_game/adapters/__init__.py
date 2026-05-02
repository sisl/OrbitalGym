"""Framework adapters — thin wrappers at the host-array boundary.

Each adapter is an opt-in extra. Import the relevant submodule; the
underlying framework (gymnasium, pettingzoo) is a peer dependency that
must be installed via the matching pip extra.

Example:
    pip install orbital-game[gymnasium]
    from orbital_game.adapters.gymnasium import GymnasiumAdapter
"""
