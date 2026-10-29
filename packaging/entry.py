"""PyInstaller entry point: the frozen binary behaves exactly like the `peepguard` command."""

from peepguard.cli import main

raise SystemExit(main())
