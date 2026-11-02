"""Provided code for Lab 4, 6.7205.4x. The public copy in codey-m/optimization (labs/helpers/) is written by
make_lab4_helpers.py from the course repository; do not edit it by hand.

Both games use one layout. A player's information sets are rows, ordered so that a parent comes before its
children. parent[i] is the row of the player's previous decision (-1 at a root) and parent_action[i] is the
action taken there. A strategy is a (rows, actions) array whose rows sum to 1 over the legal actions, and a
gradient has the same shape: entry (i, a) is the player's expected payoff from the games that end with their
own last action being a at row i, weighted by how likely the opponent and chance are to reach them.
"""
import io
import json
import time
import zipfile

import numpy as np

SNAP = 1e-12     # regret increments this close to zero, relative to the row, are ties and count as zero
SMALL = 1e-20    # below this a row's weight counts as zero (the package's own threshold)

CONFIGS = {
    "C0": dict(label="CFR", rmplus=False, alternation=False, averaging="uniform"),
    "C1": dict(label="C0 + regret matching+", rmplus=True, alternation=False, averaging="uniform"),
    "C2": dict(label="C0 + alternating updates", rmplus=False, alternation=True, averaging="uniform"),
    "C3": dict(label="C0 + linear averaging", rmplus=False, alternation=False, averaging="linear"),
    "C4": dict(label="CFR+ (all three)", rmplus=True, alternation=True, averaging="linear"),
}


# ── games ──────────────────────────────────────────────────────────────────────────────────────────────────────

class Game:
    """parent, parent_action and legal are per player; gradient(x0, x1) returns (g0, g1, ev0)."""

    def uniform(self, p):
        return self.legal[p] / self.legal[p].sum(1, keepdims=True)

    def uniform_pair(self):
        return [self.uniform(0), self.uniform(1)]

    def rows(self, p):
        return len(self.parent[p])


class Kuhn(Game):
    """Kuhn poker, as in lecture 4.3. Player 0's rows: J, Q, K, then J, Q, K after check and bet. Player 1's
    rows: J, Q, K after a check, then J, Q, K after a bet. Action 0 is check or fold, action 1 bet or call."""

    name = "Kuhn poker"
    actions = ("check/fold", "bet/call")

    def __init__(self):
        self.keys = [["J", "Q", "K", "J, check, bet", "Q, check, bet", "K, check, bet"],
                     ["J, check", "Q, check", "K, check", "J, bet", "Q, bet", "K, bet"]]
        self.parent = [np.array([-1, -1, -1, 0, 1, 2]), np.full(6, -1)]
        self.parent_action = [np.array([-1, -1, -1, 0, 0, 0]), np.full(6, -1)]
        self.legal = [np.ones((6, 2), bool), np.ones((6, 2), bool)]
        terminals = []
        for c0 in range(3):
            for c1 in range(3):
                if c0 == c1:
                    continue
                s = 1 if c0 > c1 else -1
                terminals += [(c0, 0, c1, 0, s), (3 + c0, 0, c1, 1, -1), (3 + c0, 1, c1, 1, 2 * s),
                              (c0, 1, 3 + c1, 0, 1), (c0, 1, 3 + c1, 1, 2 * s)]
        self._terminals = terminals

    @staticmethod
    def equilibrium(alpha=1 / 6):
        """Kuhn's equilibrium family for player 0 (alpha from 0 to 1/3), with player 1's equilibrium reply."""
        x0 = np.array([[1 - alpha, alpha], [1, 0], [1 - 3 * alpha, 3 * alpha],
                       [1, 0], [2 / 3 - alpha, alpha + 1 / 3], [0, 1]])
        x1 = np.array([[2 / 3, 1 / 3], [1, 0], [0, 1], [1, 0], [2 / 3, 1 / 3], [0, 1]])
        return [x0, x1]

    def gradient(self, x0, x1):
        s0, s1 = _reach(x0, self.parent[0], self.parent_action[0]), _reach(x1, self.parent[1], self.parent_action[1])
        g0, g1 = np.zeros_like(s0), np.zeros_like(s1)
        for r0, a0, r1, a1, u in self._terminals:
            g0[r0, a0] += s1[r1, a1] * u / 6
            g1[r1, a1] -= s0[r0, a0] * u / 6
        return g0, g1, float((s0 * g0).sum())


class CornerPTTT(Game):
    """Abrupt phantom tic-tac-toe with both opening moves fixed: player 0 holds the center (cell 4), player 1 a
    corner (cell 0), and both know it. Cells are numbered 0 to 8 from the top left. The game comes from
    exp-a-spiel (Gabriele Farina, MIT license) with the course's patch."""

    name = "phantom tic-tac-toe, center against corner"
    OPENING = ([4], [0])

    def __init__(self):
        import eas
        self.eas = eas
        eas.set_pttt_opening(*self.OPENING, True)
        self.traverser = eas.CornerAbruptPtttTraverser()
        x0, x1 = self.traverser.construct_uniform_strategies()
        self.parent, self.parent_action, self.legal = [], [], []
        for p, x in ((0, x0), (1, x1)):
            n = x.shape[0]
            parent, action = np.empty(n, np.int64), np.empty(n, np.int64)
            for r in range(n):
                parent[r], action[r] = self.traverser.parent_index_and_action(p, r)
            parent[0], action[0] = -1, -1          # the package marks its root row as its own parent
            self.parent.append(parent)
            self.parent_action.append(action)
            self.legal.append(x > 0)

    def gradient(self, x0, x1):
        r = self.traverser.ev_and_exploitability(np.ascontiguousarray(x0, np.float64),
                                                 np.ascontiguousarray(x1, np.float64))
        return np.asarray(r.gradient[0]), np.asarray(r.gradient[1]), float(r.ev0)

    def package_values(self, x0, x1):
        """The package's own expected value and best-response values, for checking yours."""
        r = self.traverser.ev_and_exploitability(np.ascontiguousarray(x0, np.float64),
                                                 np.ascontiguousarray(x1, np.float64))
        return dict(ev0=float(r.ev0), br0=float(r.expl[1] + r.ev0), br1=float(r.expl[0] - r.ev0))

    def infoset(self, p, row):
        return self.traverser.infoset_desc(p, row)


def _reach(x, parent, parent_action):
    sf = np.array(x, float)
    for i in range(len(sf)):
        if parent[i] >= 0:
            sf[i] *= sf[parent[i], parent_action[i]]
    return sf


def behavioral(sf, legal):
    """Turn a sequence-form strategy back into a behavioral one: normalize each row; an unreachable row is uniform."""
    s = np.where(legal, sf, 0.0).sum(1, keepdims=True)
    return np.where(s > SMALL, sf / np.where(s > SMALL, s, 1.0), legal / legal.sum(1, keepdims=True))


# ── evaluation ─────────────────────────────────────────────────────────────────────────────────────────────────

def nash_conv(game, pair, best_response_value):
    """How much the two best responders gain together: zero exactly at an equilibrium."""
    g0, g1, ev0 = game.gradient(*pair)
    br0 = best_response_value(g0, game.parent[0], game.parent_action[0], game.legal[0])
    br1 = best_response_value(g1, game.parent[1], game.parent_action[1], game.legal[1])
    return br0 + br1


def head_to_head(game, a, b):
    """Strategy pair a's average payoff per game against pair b, each playing both seats once."""
    return 0.5 * (game.gradient(a[0], b[1])[2] - game.gradient(b[0], a[1])[2])


def head_to_head_table(game, pairs):
    names = list(pairs)
    n = len(names)
    table = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            table[i, j] = head_to_head(game, pairs[names[i]], pairs[names[j]])
            table[j, i] = -table[i, j]
    field = {names[i]: float(table[i].sum() / (n - 1)) for i in range(n)}
    return names, table, field


# ── CFR ────────────────────────────────────────────────────────────────────────────────────────────────────────

class Run:
    def __init__(self, config):
        self.config = config
        self.average, self.current = {}, {}
        self.seconds_per_traversal = None


def run_cfr(game, config, budgets, regret_matching, sequence_form, counterfactual_values, quiet=False):
    """Run one configuration of CFR, keeping the average and current strategies at each budget.

    A budget counts traversals: one traversal computes both players' gradients. With alternating updates a
    traversal updates one player, so each player is updated every other traversal; without, both every time.
    The average is taken in sequence form, weighted 1 per update (uniform) or t at a player's t-th update
    (linear), starting from the uniform strategy, as in exp-a-spiel's CfrSolver.
    """
    conf = CONFIGS[config] if isinstance(config, str) else config
    budgets = sorted(set(budgets))
    x = game.uniform_pair()
    regrets = [np.zeros_like(x[0]), np.zeros_like(x[1])]
    avg = [sequence_form(x[p], game.parent[p], game.parent_action[p]) for p in (0, 1)]
    weight = [1.0, 1.0]          # the uniform start counts as the first strategy
    updates = [1, 1]
    steps = 0
    run = Run(conf)
    started = time.time()
    for t in range(1, budgets[-1] + 1):
        g = game.gradient(x[0], x[1])[:2]
        for _ in range(1 if conf["alternation"] else 2):
            p = steps % 2
            cfv = counterfactual_values(g[p], x[p], game.parent[p], game.parent_action[p])
            legal = game.legal[p]
            r = np.where(legal, cfv - (x[p] * cfv).sum(1, keepdims=True), 0.0)
            scale = np.abs(np.where(legal, cfv, 0.0)).max(1, keepdims=True)
            r[np.abs(r) <= SNAP * scale] = 0.0
            regrets[p] = regrets[p] + r
            x[p] = regret_matching(regrets[p], legal)
            if conf["rmplus"]:                     # as exp-a-spiel does: the strategy first, then the clip
                regrets[p] = np.maximum(regrets[p], 0.0)
            updates[p] += 1
            w = 1.0 if conf["averaging"] == "uniform" else float(updates[p])
            weight[p] += w
            avg[p] = avg[p] + (w / weight[p]) * (sequence_form(x[p], game.parent[p], game.parent_action[p]) - avg[p])
            steps += 1
        if t in budgets:
            run.average[t] = [behavioral(avg[p], game.legal[p]) for p in (0, 1)]
            run.current[t] = [x[0].copy(), x[1].copy()]
    run.seconds_per_traversal = (time.time() - started) / budgets[-1]
    if not quiet:
        print(f"{config if isinstance(config, str) else conf['label']}: {budgets[-1]} traversals, "
              f"{run.seconds_per_traversal:.3f} s each")
    return run


def check_against_package(game, regret_matching, sequence_form, counterfactual_values, best_response_value):
    """Your functions against exp-a-spiel's own: best responses to the uniform strategy, and plain CFR's
    average after three traversals. Returns the largest differences."""
    u = game.uniform_pair()
    g0, g1, _ = game.gradient(*u)
    pkg = game.package_values(*u)
    br = max(abs(best_response_value(g0, game.parent[0], game.parent_action[0], game.legal[0]) - pkg["br0"]),
             abs(best_response_value(g1, game.parent[1], game.parent_action[1], game.legal[1]) - pkg["br1"]))
    eas = game.eas
    solver = eas.CornerAbruptPtttCfrSolver(game.traverser, eas.CfrConf(
        avg=eas.AveragingStrategy.UNIFORM, alternation=False, dcfr=False, rmplus=False, predictive=False))
    for _ in range(3):
        solver.step()
    theirs = solver.avg_bh()
    mine = run_cfr(game, "C0", [3], regret_matching, sequence_form, counterfactual_values, quiet=True).average[3]
    cfr = max(float(np.abs(mine[p] - np.asarray(theirs[p])).max()) for p in (0, 1))
    return dict(best_response=br, cfr=cfr)


# ── submission ─────────────────────────────────────────────────────────────────────────────────────────────────

def validate(game, pair):
    """The checks the grading service runs first. Returns a list of problems; empty means valid."""
    problems = []
    for p, x in enumerate(pair):
        x = np.asarray(x)
        shape = (game.rows(p), game.legal[p].shape[1])
        if x.shape != shape:
            problems.append(f"player {p}: shape {x.shape}, expected {shape}")
            continue
        if not np.isfinite(x).all():
            problems.append(f"player {p}: values that are not finite")
            continue
        if (x < 0).any() or (x > 1).any():
            problems.append(f"player {p}: values outside 0 to 1")
        if np.abs(x.sum(1) - 1).max() > 1e-9:
            problems.append(f"player {p}: rows that do not sum to 1")
        if (x[~game.legal[p]] != 0).any():
            problems.append(f"player {p}: probability on illegal actions")
    return problems


def certify(game, pair, budget, path="lab4_submission.zip"):
    """Write the submission file, only if every check passes."""
    problems = validate(game, pair)
    if problems:
        raise ValueError("Not written: " + "; ".join(problems))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in (0, 1):
            arr = io.BytesIO()
            np.save(arr, np.asarray(pair[p], np.float64))
            z.writestr(f"pl{p}.npy", arr.getvalue())
        z.writestr("run.json", json.dumps(dict(game="corner 4-0", config="C4", budget=int(budget))))
    with open(path, "wb") as f:
        f.write(buf.getvalue())
    print(f"Wrote {path} ({len(buf.getvalue()) / 1e6:.2f} MB): the average strategy after {budget} traversals.")
    return path


# ── display ────────────────────────────────────────────────────────────────────────────────────────────────────

def show_game(moves=(1, 2, 2, 8, 7)):
    """Play a short game from the fixed opening and show what each player knows after each probe."""
    import eas
    eas.set_pttt_opening(*CornerPTTT.OPENING, True)
    s = eas.CornerAbruptPtttState()
    mark, name = "XO", ("X (player 0)", "O (player 1)")
    held = [{4}, {0}]                       # the true board
    known = [{0}, {4}]                      # opponent cells each player knows about
    print("Opening: X holds the center (cell 4) and O the top-left corner (cell 0), and both know it.")
    print("Cells are numbered 0 to 8 from the top left. A probe of a taken cell fails, and the turn passes.\n")
    for cell in moves:
        if s.is_terminal():
            break
        p = s.player()
        seen = s.infoset_desc()
        landed = cell not in held[1 - p]
        s.next(cell)
        (held[p] if landed else known[p]).add(cell)
        outcome = "placed" if landed else f"already {mark[1 - p]}'s, so now {mark[p]} knows it"
        print(f"{name[p]} has seen {seen!r} and probes cell {cell}: {outcome}.")
    if s.is_terminal():
        w = s.winner()
        print("\n" + ("A draw." if w is None else f"{mark[w]} has three in a row and wins."))
    rows = [_board(held[0], held[1]), _board(held[0], known[0] & held[1]), _board(known[1] & held[0], held[1])]
    print("\n  true board   X's view   O's view")
    for line in zip(*rows):
        print("     " + "        ".join(line))
    print("\nIn an information-set string, '4*' means 'I placed a mark on cell 4' and '2.' means 'cell 2 was taken'.")


def _board(xs, os_):
    cells = ["X" if c in xs else "O" if c in os_ else "." for c in range(9)]
    return ["".join(cells[r * 3:r * 3 + 3]) for r in range(3)]


def print_ladder(runs, game, best_response_value, current=False):
    """NashConv of each run's average strategy at each budget, the current strategy's in brackets if asked."""
    budgets = sorted(next(iter(runs.values())).average)
    print(f"{'NashConv':34}" + "".join(f"{t:>16}" for t in budgets) + f"{'s per traversal':>18}")
    for name, run in runs.items():
        cells = []
        for t in budgets:
            a = nash_conv(game, run.average[t], best_response_value)
            c = f" ({nash_conv(game, run.current[t], best_response_value):.3f})" if current else ""
            cells.append(f"{a:.4f}{c}")
        print(f"{name + ' ' + run.config['label']:34}" + "".join(f"{s:>16}" for s in cells)
              + f"{run.seconds_per_traversal:>18.3f}")


def print_head_to_head(names, field, nash):
    """Strategies ranked by their mean head-to-head result against the others, with each one's NashConv.

    Two columns rescale to net points per 100 games, a win counting +1, a loss -1 and a draw 0: the strategy's own
    result against the field, and a best responder's result against the strategy over 100 games, 50 in each seat.
    The second is 50 x NashConv, because NashConv adds the best responder's gains in the two seats."""
    by_nash = {n: r for r, n in enumerate(sorted(names, key=lambda n: nash[n]), 1)}
    print(f"{'':16}{'against the field,':>20}{'':10}{'best responder,':>18}")
    print(f"{'rank':>4}  {'strategy':10}{'per 100 games':>20}{'NashConv':>10}{'per 100 games':>18}{'rank by NashConv':>18}")
    for r, n in enumerate(sorted(names, key=lambda n: -field[n]), 1):
        print(f"{r:>4}  {n:10}{100 * field[n]:>+20.2f}{nash[n]:>10.4f}{50 * nash[n]:>+18.2f}{by_nash[n]:>18}")


def plot_runs(runs, game, best_response_value, title):
    """NashConv of the average (solid) and current (dashed) strategy at each budget, one color per run."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for (name, run), color in zip(runs.items(), plt.rcParams["axes.prop_cycle"].by_key()["color"]):
        ts = sorted(run.average)
        ax.plot(ts, [nash_conv(game, run.average[t], best_response_value) for t in ts], "-o", color=color,
                label=f"{name} average")
        ax.plot(ts, [nash_conv(game, run.current[t], best_response_value) for t in ts], "--", color=color,
                label=f"{name} current")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("traversals")
    ax.set_ylabel("NashConv (lower is better)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.show()
