"""
Progress Tracking Utilities

Provides progress bars and ETA for long-running operations.
"""

from typing import Optional, Iterator, Any
from contextlib import contextmanager

try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class ProgressTracker:
    """
    Context manager for progress tracking with fallback.

    Example:
        >>> with ProgressTracker(total=100, desc="Processing") as pbar:
        ...     for i in range(100):
        ...         # Do work
        ...         pbar.update(1)
    """

    def __init__(
        self,
        total: Optional[int] = None,
        desc: Optional[str] = None,
        disable: bool = False,
        unit: str = "it"
    ):
        """
        Args:
            total: Total number of iterations
            desc: Description shown in progress bar
            disable: If True, disable progress bar
            unit: Unit name (e.g., "prompt", "sample")
        """
        self.total = total
        self.desc = desc
        self.disable = disable or not TQDM_AVAILABLE
        self.unit = unit
        self.pbar = None
        self.current = 0

    def __enter__(self):
        if not self.disable and TQDM_AVAILABLE:
            self.pbar = tqdm(
                total=self.total,
                desc=self.desc,
                unit=self.unit,
                leave=True
            )
        return self

    def __exit__(self, *args):
        if self.pbar is not None:
            self.pbar.close()

    def update(self, n: int = 1):
        """Update progress by n steps"""
        self.current += n
        if self.pbar is not None:
            self.pbar.update(n)
        elif not self.disable:
            # Fallback: print every 10%
            if self.total and self.current % max(1, self.total // 10) == 0:
                pct = (self.current / self.total) * 100
                print(f"  {self.desc}: {self.current}/{self.total} ({pct:.0f}%)")

    def set_postfix(self, **kwargs):
        """Set postfix values (e.g., loss, accuracy)"""
        if self.pbar is not None:
            self.pbar.set_postfix(**kwargs)

    def set_description(self, desc: str):
        """Update description"""
        self.desc = desc
        if self.pbar is not None:
            self.pbar.set_description(desc)


def track_progress(
    iterable: Iterator[Any],
    desc: Optional[str] = None,
    total: Optional[int] = None,
    disable: bool = False,
    unit: str = "it"
) -> Iterator[Any]:
    """
    Wrap iterable with progress bar.

    Example:
        >>> for item in track_progress(items, desc="Processing"):
        ...     process(item)

    Args:
        iterable: Iterable to wrap
        desc: Description
        total: Total length (auto-detected if possible)
        disable: Disable progress bar
        unit: Unit name

    Yields:
        Items from iterable
    """
    if disable or not TQDM_AVAILABLE:
        # Fallback: just iterate
        for i, item in enumerate(iterable):
            if total and i % max(1, total // 10) == 0:
                pct = (i / total) * 100
                print(f"  {desc}: {i}/{total} ({pct:.0f}%)")
            yield item
    else:
        yield from tqdm(iterable, desc=desc, total=total, unit=unit)


@contextmanager
def progress_section(desc: str, disable: bool = False):
    """
    Context manager for a named progress section.

    Example:
        >>> with progress_section("Loading model"):
        ...     model = load_model()
        ✓ Loading model (2.3s)

    Args:
        desc: Section description
        disable: Disable output
    """
    import time

    if not disable:
        print(f"\n{desc}...", end='', flush=True)

    start_time = time.time()

    try:
        yield
        elapsed = time.time() - start_time
        if not disable:
            print(f" ✓ ({elapsed:.1f}s)")
    except Exception as e:
        if not disable:
            print(f" ✗ Failed")
        raise
