"""Fault separator contract of banked joint extraction (``joint`` with one planar fault).

Each bank is extracted separately with the fault as a *separator*: its field is
bank-normalized so the excluded side is positive and the separator level is
zero, and it owns its affected surfaces as an ordinary controller. The
separator must be affine with constant actual gradients.
"""

import numpy as np

from ...core.data.stack_relation_type import StackRelationType as R


def check_separator_contract(separator_contacts, surface_to_stack, stack_relations, fault_pairs):
    """Validate ``separator_contacts``; return ``(separator, bank)``.

    It needs exactly the keys separator, bank, fault_pairs and controllers;
    ``fault_pairs`` must equal the represented directed fault edges, all from
    the separator, which must be the only fault stack.
    """
    if not isinstance(separator_contacts, dict) or set(separator_contacts) != {
            'separator', 'bank', 'fault_pairs', 'controllers'}:
        raise ValueError('invalid_separator_contract: explicit separator, bank, pairs and controllers required')
    separator, bank = separator_contacts['separator'], separator_contacts['bank']
    n = len(surface_to_stack)
    if (not isinstance(separator, (int, np.integer)) or isinstance(separator, bool) or not 0 <= separator < n
            or not isinstance(bank, (int, np.integer)) or isinstance(bank, bool) or bank not in (0, 1)):
        raise ValueError('invalid_separator_contract: invalid local separator or bank')
    supplied = separator_contacts['fault_pairs']
    if not isinstance(supplied, set) or not supplied:
        raise ValueError('unauthorized_fault_pairs: require exactly the represented directed fault edges')
    if any(not isinstance(pair, tuple) or len(pair) != 2 or any(
            not isinstance(index, (int, np.integer)) or isinstance(index, bool) or not 0 <= index < n
            for index in pair) for pair in supplied):
        raise ValueError('invalid_fault_pairs: integer nonboolean in-bounds local source/target indices required')
    if supplied != fault_pairs or any(
            c != separator or c == t or surface_to_stack[c] == surface_to_stack[t] for c, t in supplied):
        raise ValueError('unauthorized_fault_pairs: require exactly the represented directed fault edges')
    if stack_relations[surface_to_stack[separator]] is not R.FAULT:
        raise ValueError('invalid_separator_contract: separator must belong to a fault stack')
    if [g for g, relation in enumerate(stack_relations) if relation is R.FAULT] != [surface_to_stack[separator]]:
        raise ValueError('unsupported_multiple_faults: full metadata must contain only the represented separator fault')
    if not isinstance(separator_contacts['controllers'], dict):
        raise ValueError('unsupported_ownership: explicit composite controller dict required')
    return int(separator), int(bank)


def authorize_fault_pairs(allowed, fault_pairs):
    """Eligibility with the authorized separator pairs added (both directions)."""
    allowed = allowed.copy()
    for c, t in fault_pairs:
        allowed[c, t] = allowed[t, c] = True
    return allowed


def separator_controllers(controllers, fault_pairs):
    """The separator owns its targets' negative (non-excluded) side."""
    controllers = {t: list(rows) for t, rows in controllers.items()}
    for c, t in sorted(fault_pairs):
        controllers[t].append((c, -1))
    return controllers


def fit_separator_plane(points, raw, gradients):
    """``(gradient, slope, offset)`` of an affine separator with constant aligned actual gradients."""
    gradient = gradients[0]
    slope = np.linalg.lstsq(points-points[0], raw-raw[0], rcond=None)[0]
    offset = raw[0] - points[0] @ slope
    if (np.linalg.norm(gradient) < 1e-12 or np.linalg.norm(slope) < 1e-12
            or not np.allclose(gradients, gradient, atol=1e-10, rtol=0)
            or not np.allclose(raw, points @ slope + offset, atol=1e-10, rtol=0)):
        raise ValueError('unsupported_separator_geometry: canonical field must be affine with constant actual gradients')
    if gradient @ slope < (1-1e-10)*np.linalg.norm(gradient)*np.linalg.norm(slope):
        raise ValueError('unsupported_separator_orientation: actual gradient must align with normalized affine field')
    return gradient.copy(), slope, offset


def check_separator_samples(points, raw, gradients, plane):
    """Later samples must stay on the fitted separator plane."""
    gradient, slope, offset = plane
    if not (np.allclose(gradients, gradient, atol=1e-10, rtol=0)
            and np.allclose(raw, points @ slope + offset, atol=1e-10, rtol=0)):
        raise ValueError('unsupported_separator_geometry: canonical field must be affine with constant actual gradients')


def bank_normalized(values, gradients, rows, sign, level):
    """Separator rows as ``sign * (raw - level)`` with gradients oriented by ``sign``."""
    values, gradients = values.copy(), gradients.copy()
    values[rows] = sign*(values[rows]-level)
    gradients[rows] *= sign
    return values, gradients
