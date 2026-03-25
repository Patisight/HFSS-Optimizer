#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试导入"""
import sys
sys.path.insert(0, '.')

print("Testing imports...")

try:
    from algorithms.mopso import MOPSO
    print("[OK] MOPSO")
except Exception as e:
    print(f"[ERROR] MOPSO: {e}")

try:
    from algorithms.mobo import MOBO
    print("[OK] MOBO")
except Exception as e:
    print(f"[ERROR] MOBO: {e}")

try:
    from algorithms.nsga2 import NSGA2
    print("[OK] NSGA2")
except Exception as e:
    print(f"[ERROR] NSGA2: {e}")

try:
    from core.surrogate import SurrogateModel, SurrogateManager
    print("[OK] Surrogate")
except Exception as e:
    print(f"[ERROR] Surrogate: {e}")

print("\nAll imports tested!")
