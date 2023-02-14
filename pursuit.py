import time
import numpy as np
import scipy.sparse as sps
import scipy.optimize
import matplotlib.pyplot as plt
import tqdm
from functools import partial


def _treshold(v, k: int):
    av = np.abs(v)
    t = np.sort(av)[-k]  # get the kth largest value
    res = v.copy()
    res[av < t] = 0
    return res


def _support(v, k: int):
    av = np.abs(v)
    return np.argpartition(-av, k)[:k]


def lp(M, x, s: int = None):
    # Minimize ||z||_1 such that Xz = x
    k, n = M.shape
    res = scipy.optimize.linprog(
        # We use extra variables y_1, ..., y_n such that y_i = |z_i|.
        # We stack the variables [z_1, ..., z_n, y_1, ..., y_n].
        c=np.concatenate([np.zeros(n), np.ones(n)]),
        # y >= z and y >= -z equiv. z - y <= 0 and -z - y <= 0.
        A_ub=np.block([[np.eye(n), -np.eye(n)], [-np.eye(n), -np.eye(n)]]),
        # Using sparse matrices should be faster, but it isn't for our dim=100
        # A_ub=sps.bmat([[sps.eye(n), -sps.eye(n)], [-sps.eye(n), -sps.eye(n)]]),
        b_ub=np.zeros(2 * n),
        A_eq=np.hstack([M, np.zeros((k, n))]),
        b_eq=x,
        bounds=(None, None),
    )
    if res.x is None:
        print(f"Warning: No LP result with {k=} and {s=}.")
        return np.zeros(n)
    z = res.x[:n]
    if s is not None:
        z = _treshold(z, s)
    return z


def omp(M, x, s: int):
    k, n = M.shape
    # Normalize columns
    norms = np.linalg.norm(M, axis=0, keepdims=True)
    M2 = M / norms
    residual, ids = x, []
    # Greedily select the best columns to explain x
    for _ in range(s):
        ids.append(np.argmax(np.abs(M2.T @ residual)))
        m, _, _, _ = np.linalg.lstsq(M2[:, ids], x, rcond=None)
        residual = x - M2[:, ids] @ m
    # Return the result as an m-sparse vector
    res = np.zeros(n)
    res[ids] = m
    return res / norms[0]


def cosamp(M, x, s: int, max_iter=10, rtol=1e-5, atol=1e-8, pinv=False):
    k, n = M.shape
    norms = np.linalg.norm(M, axis=0, keepdims=True)
    M2 = M / norms
    MPI = np.linalg.pinv(M2) if pinv else M2.T

    z, residual = np.zeros(n), x
    # Some large initial number to measure progress
    old_res_mean = 100
    for it in range(max_iter):
        # Form the ids set as the union of the previous support
        # and the 2s largest indices in the signal proxy
        ids = list(set(_support(z, s)) | set(_support(MPI @ residual, 2 * s)))
        m, _, _, _ = np.linalg.lstsq(M2[:, ids], x, rcond=None)

        # Map coefficients back to the full space
        z = np.zeros(n)
        z[ids] = m
        z = _treshold(z, s)
        residual = x - M2 @ z

        # Compute loss
        res_mean = (residual**2).mean()
        if np.isclose(res_mean, old_res_mean, rtol=rtol, atol=atol):
            break
        old_res_mean = res_mean
    return z / norms[0]


def cosamp2(M, x, s: int, max_iter=10, pinv=False):
    # A different implementation of CoSaMP that uses convergence in the
    # selected subspace as the suppoing condition instead of numerical tolerances.

    k, n = M.shape
    norms = np.linalg.norm(M, axis=0, keepdims=True)
    M2 = M / norms
    MPI = np.linalg.pinv(M2) if pinv else M2.T

    ids, residual = list(_support(MPI @ x, 2 * s)), x
    for it in range(max_iter):
        m, _, _, _ = np.linalg.lstsq(M2[:, ids], x, rcond=None)

        # Trim back down to "s" many ids
        m_ids = _support(m, s)
        m, trimmed_ids = m[m_ids], np.array(ids)[m_ids]

        # Form the ids set as the union of the previous support
        # and the 2s largest indices in the signal proxy
        residual = x - M2[:, trimmed_ids] @ m
        new_ids = sorted(set(trimmed_ids) | set(_support(MPI @ residual, 2 * s)))

        # Test for convergence
        if new_ids == ids:
            break
        ids = new_ids

    # Map coefficients back to the full space
    z = np.zeros(n)
    z[trimmed_ids] = m
    return z / norms[0]


def iht(M, x, s: int, max_iter=20, pinv=False):
    k, n = M.shape
    norms = np.linalg.norm(M, axis=0, keepdims=True)
    M2 = M / norms
    MPI = np.linalg.pinv(M2) if pinv else M2.T
    y = np.zeros(n)
    for it in range(max_iter):
        grad = M2 @ y - x
        y = _treshold(y - MPI @ grad, s)
    return y / norms[0]


def iht2(M, x, s: int, max_iter=20):
    k, n = M.shape
    norms = np.linalg.norm(M, axis=0, keepdims=True)
    M2 = M / norms
    y = np.zeros(n)
    for it in range(max_iter):
        g = M2.T @ (M2 @ y - x)
        ids = _support(y, s)
        mu = (g[ids] ** 2).sum() / (((M2[:,ids] @ g[ids]) ** 2).sum() + 1e-9)
        y = _treshold(y - mu * g, s)
    return y / norms[0]


def iht_ada(M, x, s: int, max_iter=20):
    # I can't really get this to work.
    k, n = M.shape
    norms = np.linalg.norm(M, axis=0, keepdims=True)
    M2 = M / norms
    y = np.zeros(n)
    g2 = 0
    for it in range(max_iter):
        g = M2.T @ (M2 @ y - x)
        g2 += g**2
        y = _treshold(y - g / np.sqrt(1e-9 + g2), s)
    return y / norms[0]


def measure(method, d, sparsity, n_measures, reps=1000):
    # Generate some measuring matrices
    Ms = np.random.randn(reps, n_measures, d)
    Ms /= np.linalg.norm(Ms, axis=1, keepdims=True)
    # And some sparse vectors to measure.
    xs = np.zeros((reps, d))
    for x in xs:
        x[:sparsity] = 1 - 2 * np.random.randint(2, size=(sparsity,))
    # We can also add some noise to the measurements, that makes sense
    # when we are trying to measure the recovery error, but it makes it
    # harder to measure the recovery rate, so for now we just use zero noise.
    # noises = np.random.randn(reps, n_measures) / d**.5
    noises = np.zeros((reps, n_measures))
    # Precompute the measurements so they don't influence benchmarking
    Mxs = [M @ x + nois for M, x, nois in zip(Ms, xs, noises)]
    err2 = 0
    recp = 0
    t = time.time()
    for M, Mx, x in zip(Ms, Mxs, xs):
        z = method(M, Mx, sparsity)
        success = np.abs(z - x).sum() < 1e-1
        recp += int(success)
        err2 += np.sum((Mx - M @ z) ** 2)
    elapsed = time.time() - t
    recp /= reps
    err = np.sqrt(err2 / reps)
    return err, recp, elapsed


def main():
    n_sparsity = 10
    n_measurements = 50
    dim = 100
    reps = 100
    methods = [lp, omp, cosamp2, iht, iht2, partial(iht, pinv=True)]
    titles = ["LinProg", "OMP", "CoSaMP", "IHT", "IHT 2", "IHT (pinv)"]
    #methods = [iht, iht2, partial(iht, pinv=True), cosamp]
    #titles = ['iht', 'iht2', 'iht (pinv)', 'cosamp']
    assert len(methods) == len(titles)
    fig, axs = plt.subplots(1, len(methods))
    fig.suptitle(f"Exact Recovery Rate with {dim} Dimensional Gaussian Measurements")
    for ax, method, title in zip(axs, methods, titles):
        data = np.zeros((n_measurements, n_sparsity))
        elapsed = 0
        for sp in tqdm.tqdm(range(n_sparsity)):
            for mes in tqdm.tqdm(range(n_measurements), leave=False):
                _err, recp, inner_elapsed = measure(
                    method, dim, sp + 1, mes + 1, reps=reps
                )
                data[mes, sp] = recp
                elapsed += inner_elapsed
        ax.set_title(f"{title},\nt={elapsed:.1f}s")
        im = ax.imshow(
            data,
            cmap="gray",
            origin="lower",
            extent=[1, n_sparsity + 1, 1, n_measurements + 1],
        )
        ax.set_xlabel("Sparsity")
    axs[0].set_ylabel("Measurements")
    fig.colorbar(im, ax=axs.ravel().tolist())
    plt.show()


def main2():
    sp = 3
    dim = 40
    n_measurements = 20
    M = np.random.randn(n_measurements, dim)
    x = np.zeros(dim)
    x[:sp] = 1
    z = iht(M, M @ x, sp)
    print(z)
    print("z l1", np.abs(z).sum())
    print("l1 dif", np.abs(z - x).sum())
    print(M @ z - M @ x)


if __name__ == "__main__":
    main()

