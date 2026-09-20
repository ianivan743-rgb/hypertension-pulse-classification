"""训练/验证/测试划分. 支持按 group (受试者原始 sample_id) 分组, 避免同一人的 seg 泄漏."""
import numpy as np
from sklearn.model_selection import (StratifiedKFold, train_test_split,
                                     StratifiedGroupKFold, GroupShuffleSplit)


def _subject_id(sample_id: str) -> str:
    return str(sample_id).split("#", 1)[0]


def make_groups(sample_ids):
    return np.array([_subject_id(s) for s in sample_ids])


def outer_split(y, test_ratio=0.15, seed=42, groups=None):
    """留出独立测试集; 有 group 时按受试者切, 保证同一受试者不跨 train/test."""
    idx = np.arange(len(y))
    if groups is None:
        tr, te = train_test_split(idx, test_size=test_ratio, stratify=y, random_state=seed)
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        tr, te = next(gss.split(idx, y, groups=groups))
    return tr, te


def stratified_folds(y, n_splits=5, seed=42, groups=None):
    """5-fold. 有 group 时用 StratifiedGroupKFold."""
    if groups is None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, va in skf.split(np.zeros(len(y)), y):
            yield tr, va
    else:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, va in sgkf.split(np.zeros(len(y)), y, groups=groups):
            yield tr, va
