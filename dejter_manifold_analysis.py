from collections import Counter, deque
from cmath import sqrt
from math import isclose


def syndrome(x: int) -> tuple[int, int, int]:
    # Columns are binary representations of 1..7.
    # For vector bits x0..x6, syndrome is XOR-sum of selected columns.
    s0 = s1 = s2 = 0
    for j in range(7):
        if (x >> j) & 1:
            col = j + 1
            s0 ^= (col >> 0) & 1
            s1 ^= (col >> 1) & 1
            s2 ^= (col >> 2) & 1
    return s0, s1, s2


def popcount(x: int) -> int:
    return x.bit_count()


def build_graph() -> tuple[list[int], list[list[int]], list[list[float]]]:
    codewords = {x for x in range(128) if syndrome(x) == (0, 0, 0)}
    vertices = [x for x in range(128) if x not in codewords]
    idx = {v: i for i, v in enumerate(vertices)}
    n = len(vertices)

    adj_list = [[] for _ in range(n)]
    A = [[0.0] * n for _ in range(n)]

    for i, v in enumerate(vertices):
        for b in range(7):
            w = v ^ (1 << b)
            if w in idx:
                j = idx[w]
                if i < j:
                    adj_list[i].append(j)
                    adj_list[j].append(i)
                    A[i][j] = 1.0
                    A[j][i] = 1.0

    return vertices, adj_list, A


def is_bipartite(adj_list: list[list[int]]) -> bool:
    n = len(adj_list)
    color = [-1] * n
    for s in range(n):
        if color[s] != -1:
            continue
        color[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj_list[u]:
                if color[v] == -1:
                    color[v] = 1 - color[u]
                    q.append(v)
                elif color[v] == color[u]:
                    return False
    return True


def jacobi_eigenvalues(matrix: list[list[float]], tol: float = 1e-12, max_iter: int = 200000) -> list[float]:
    n = len(matrix)
    A = [row[:] for row in matrix]

    for _ in range(max_iter):
        p, q = 0, 1
        max_off = 0.0
        for i in range(n - 1):
            row = A[i]
            for j in range(i + 1, n):
                val = abs(row[j])
                if val > max_off:
                    max_off = val
                    p, q = i, j
        if max_off < tol:
            break

        app = A[p][p]
        aqq = A[q][q]
        apq = A[p][q]

        if isclose(apq, 0.0, abs_tol=tol):
            continue

        tau = (aqq - app) / (2.0 * apq)
        t = 1.0 / (abs(tau) + (tau * tau + 1.0) ** 0.5)
        if tau < 0:
            t = -t
        c = 1.0 / (1.0 + t * t) ** 0.5
        s = t * c

        for k in range(n):
            if k != p and k != q:
                akp = A[k][p]
                akq = A[k][q]
                A[k][p] = c * akp - s * akq
                A[p][k] = A[k][p]
                A[k][q] = s * akp + c * akq
                A[q][k] = A[k][q]

        A[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        A[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        A[p][q] = 0.0
        A[q][p] = 0.0

    return [A[i][i] for i in range(n)]


def main() -> None:
    d = 6
    q = d - 1
    vertices, adj_list, A = build_graph()
    n = len(vertices)

    degrees = [len(nei) for nei in adj_list]
    regular = all(deg == d for deg in degrees)
    bip = is_bipartite(adj_list)
    edges = sum(degrees) // 2

    evals = sorted(jacobi_eigenvalues(A))
    rounded = [int(round(x)) for x in evals]
    spectrum = Counter(rounded)

    print("=== Dejter Manifold (112-vertex induced subgraph of Q7) ===")
    print(f"Graph vertices |V|: {n}")
    print(f"Graph edges |E|: {edges}")
    print(f"Regularity check (degree = 6): {regular}")
    print(f"Degree min/max: {min(degrees)}/{max(degrees)}")
    print(f"Bipartite check: {bip}")

    print("\n=== Exact Adjacency Spectrum [eigenvalue : multiplicity] ===")
    for lam in sorted(spectrum):
        print(f"{lam:>3} : {spectrum[lam]}")

    print("\n=== Bass Quadratic Roots for Non-Backtracking Spectrum ===")
    print("For each adjacency eigenvalue λ, solve μ^2 - λμ + 5 = 0")

    nontrivial_mags = []
    for lam in sorted(spectrum):
        disc = lam * lam - 4 * q
        root = sqrt(disc)
        mu1 = (lam + root) / 2
        mu2 = (lam - root) / 2
        mult = spectrum[lam]
        print(
            f"λ={lam:>3} (mult={mult:>2}) -> "
            f"μ1={mu1.real:+.12f}{mu1.imag:+.12f}i, "
            f"μ2={mu2.real:+.12f}{mu2.imag:+.12f}i"
        )
        for mu in (mu1, mu2):
            mag = abs(mu)
            if not (isclose(mag, 1.0, rel_tol=0.0, abs_tol=1e-9) or isclose(mag, q, rel_tol=0.0, abs_tol=1e-9)):
                nontrivial_mags.extend([mag] * mult)

    critical = q ** 0.5
    all_on_circle = all(isclose(m, critical, rel_tol=0.0, abs_tol=1e-9) for m in nontrivial_mags)
    unique_mags = sorted({round(m, 12) for m in nontrivial_mags})

    print("\n=== Riemann Hypothesis / Ramanujan Test ===")
    print(f"Critical circle radius sqrt(q)=sqrt(5)={critical:.12f}")
    print(f"Non-trivial |μ| values (unique, rounded): {unique_mags}")
    print(f"All non-trivial |μ| on critical circle? {all_on_circle}")


if __name__ == "__main__":
    main()
