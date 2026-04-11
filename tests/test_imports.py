#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试导入"""

import sys

sys.path.insert(0, ".")


def test_import_mopso():
    """测试 MOPSO 导入"""
    from algorithms.mopso import MOPSO

    assert MOPSO is not None


def test_import_mobo():
    """测试 MOBO 导入"""
    from algorithms.mobo import MultiObjectiveBayesianOptimizer

    assert MultiObjectiveBayesianOptimizer is not None


def test_import_nsga2():
    """测试 NSGA2 导入"""
    from algorithms.nsga2 import NSGA2

    assert NSGA2 is not None


def test_import_surrogate():
    """测试 Surrogate 导入"""
    from core.surrogate import SurrogateManager, SurrogateModel

    assert SurrogateManager is not None
    assert SurrogateModel is not None
