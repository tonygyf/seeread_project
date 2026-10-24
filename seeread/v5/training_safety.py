"""Abort or skip invalid aggregate norms before any optimizer update."""
import torch


def record_amp_rejection(state, maximum, event, **details):
    """Count all overflows, but stop only on consecutive rejected updates."""
    state['amp_skips'] = state.get('amp_skips', 0) + 1
    state['amp_skip_streak'] = state.get('amp_skip_streak', 0) + 1
    event('amp_skip', count=state['amp_skips'], streak=state['amp_skip_streak'],
          maximum=maximum, **details)
    if state['amp_skip_streak'] >= maximum:
        raise FloatingPointError('Consecutive AMP rejection allowance exhausted')


def reject_nonfinite_norm(norm, *, amp, scaler, optimizer, state, event, max_skips):
    if torch.isfinite(norm):
        return False
    if not amp:
        raise FloatingPointError('Nonfinite reading gradients')
    # unscale_ only checks individual gradient entries. A norm reduction can
    # overflow later even when every entry was finite.
    scaler.update(new_scale=scaler.get_scale() / 2)
    optimizer.zero_grad(set_to_none=True)
    scaled = scaler.state_dict()
    scaled['_growth_tracker'] = 0
    scaler.load_state_dict(scaled)
    record_amp_rejection(state, max_skips, event, scale=scaler.get_scale(), reason='nonfinite_gradient_norm')
    return True
