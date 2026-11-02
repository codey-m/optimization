"""Smoke test for a built wheel: the lab's game has the known size and uniform NashConv."""
import eas

eas.set_pttt_opening([4], [0], True)
t = eas.CornerAbruptPtttTraverser()
assert (t.NUM_INFOS_PL0, t.NUM_INFOS_PL1) == (46824, 29435), (t.NUM_INFOS_PL0, t.NUM_INFOS_PL1)
x, y = t.construct_uniform_strategies()
r = t.ev_and_exploitability(x, y)
nash_conv = sum(r.expl)
assert abs(nash_conv - 1.2720238095238094) < 1e-9, nash_conv
print(f"corner 4-0: {t.NUM_INFOS_PL0} / {t.NUM_INFOS_PL1} rows, uniform NashConv {nash_conv:.10f}")
