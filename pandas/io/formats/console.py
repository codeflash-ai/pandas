"""
Internal module for console introspection
"""

from __future__ import annotations

from shutil import get_terminal_size


def get_console_size() -> tuple[int | None, int | None]:
    """
    Return console size as tuple = (width, height).

    Returns (None,None) in non-interactive session.
    """
    # Only import if needed
    from pandas import get_option

    # Fetch options once; check for None (auto-detection)
    display_width = get_option("display.width")
    display_height = get_option("display.max_rows")

    # Localize auto-detection decisions
    auto_width = display_width is None
    auto_height = display_height is None

    # decide if need to auto-detect
    if (auto_width or auto_height) and in_interactive_session():
        if in_ipython_frontend():
            # Sane defaults for interactive non-shell terminal
            from pandas._config.config import get_default_val

            terminal_width = get_default_val("display.width")
            terminal_height = get_default_val("display.max_rows")
        else:
            terminal_width, terminal_height = get_terminal_size()
        width = display_width if not auto_width else terminal_width
        height = display_height if not auto_height else terminal_height
    else:
        # Non-interactive or direct values just use config or None
        width = display_width
        height = display_height

    return width, height


# ----------------------------------------------------------------------
# Detect our environment


def in_interactive_session() -> bool:
    """
    Check if we're running in an interactive shell.

    Returns
    -------
    bool
        True if running under python/ipython interactive shell.
    """
    from pandas import get_option

    def check_main() -> bool:
        try:
            import __main__ as main
        except ModuleNotFoundError:
            return get_option("mode.sim_interactive")
        return not hasattr(main, "__file__") or get_option("mode.sim_interactive")

    try:
        # error: Name '__IPYTHON__' is not defined
        return __IPYTHON__ or check_main()  # type: ignore[name-defined]
    except NameError:
        return check_main()


def in_ipython_frontend() -> bool:
    """
    Check if we're inside an IPython zmq frontend.

    Returns
    -------
    bool
    """
    try:
        # error: Name 'get_ipython' is not defined
        ip = get_ipython()  # type: ignore[name-defined]
        return "zmq" in str(type(ip)).lower()
    except NameError:
        pass

    return False
