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


def omp(M, x, s: int, pinv=False):
    k, n = M.shape
    MPI = np.linalg.pinv(M) if pinv else M.T
    # Normalize columns
    residual, ids = x, []
    # Greedily select the best columns to explain x
    for _ in range(s):
        ids.append(np.argmax(np.abs(MPI @ residual)))
        m, _, _, _ = np.linalg.lstsq(M[:, ids], x, rcond=None)
        residual = x - M[:, ids] @ m
    # Return the result as an m-sparse vector
    res = np.zeros(n)
    res[ids] = m
    return res


def omp2(M, x, s: int):
    # Try doing two iterations of OMP, where the second one allows
    # replacing the indicies found in the first round with new ones.
    # Conclusion: Doesn't seem to work very well.

    k, n = M.shape
    # Normalize columns
    residual, ids = x, []

    # Greedily select the best columns to explain x
    for _ in range(2*s+1):
        ids.append(np.argmax(np.abs(M.T @ residual)))
        if len(ids) == s + 1:
            del ids[0]
        m, _, _, _ = np.linalg.lstsq(M[:, ids], x, rcond=None)
        residual = x - M[:, ids] @ m
    # Return the result as an m-sparse vector
    res = np.zeros(n)
    res[ids] = m
    return res


def cosamp(M, x, s: int, max_iter=10, rtol=1e-5, atol=1e-8, pinv=False):
    k, n = M.shape
    MPI = np.linalg.pinv(M) if pinv else M.T

    z, residual = np.zeros(n), x
    # Some large initial number to measure progress
    old_res_mean = 100
    for it in range(max_iter):
        # Form the ids set as the union of the previous support
        # and the 2s largest indices in the signal proxy
        ids = list(set(_support(z, s)) | set(_support(MPI @ residual, 2 * s)))
        m, _, _, _ = np.linalg.lstsq(M[:, ids], x, rcond=None)

        # Map coefficients back to the full space
        z = np.zeros(n)
        z[ids] = m
        z = _treshold(z, s)
        residual = x - M @ z

        # Compute loss
        res_mean = (residual**2).mean()
        if np.isclose(res_mean, old_res_mean, rtol=rtol, atol=atol):
            break
        old_res_mean = res_mean
    return z


def cosamp2(M, x, s: int, max_iter=10, pinv=False):
    # A different implementation of CoSaMP that uses convergence in the
    # selected subspace as the suppoing condition instead of numerical tolerances.

    k, n = M.shape
    MPI = np.linalg.pinv(M) if pinv else M.T

    ids, residual = list(_support(MPI @ x, 2 * s)), x
    for it in range(max_iter):
        m, _, _, _ = np.linalg.lstsq(M[:, ids], x, rcond=None)

        # Trim back down to "s" many ids
        m_ids = _support(m, s)
        m, trimmed_ids = m[m_ids], np.array(ids)[m_ids]

        # Form the ids set as the union of the previous support
        # and the 2s largest indices in the signal proxy
        residual = x - M[:, trimmed_ids] @ m
        new_ids = sorted(set(trimmed_ids) | set(_support(MPI @ residual, 2 * s)))

        # Test for convergence
        if new_ids == ids:
            break
        ids = new_ids

    # Map coefficients back to the full space
    z = np.zeros(n)
    z[trimmed_ids] = m
    return z


def cosamp3(M, x, s: int, max_iter=10, pinv=False):
    # This version is a tiny bit worse than cosamp2, but a lot faster.
    # This is because it always uses a draft subspace of size 2s, rather than
    # somewhere between 2s and 3s.

    # If IHT keeps doing:
    #    grad = MPI @ (M @ y - x)
    #    y -= grad
    #    y = _treshold(y, s)
    # This function keeps doing:
    #    grad = MPI @ (M @ y - x)
    #    ids = _support(y - grad, 2s)
    #    y = project x into M[ids]
    #    y = _treshold(y, s)

    k, n = M.shape
    MPI = np.linalg.pinv(M) if pinv else M.T

    ids, residual = list(_support(MPI @ x, 2 * s)), x
    for it in range(max_iter):
        m, _, _, _ = np.linalg.lstsq(M[:, ids], x, rcond=None)

        # Trim back down to "s" many ids
        m_ids = _support(m, s)
        m, trimmed_ids = m[m_ids], np.array(ids)[m_ids]

        z = np.zeros(n)
        z[trimmed_ids] = m
        # Instead of taking the union of the supports, we just add the vectors
        # and compute the support of the result
        residual = x - M[:, trimmed_ids] @ m
        new_ids = list(_support(z + MPI @ residual, 2 * s))

        # Test for convergence
        if new_ids == ids:
            break
        ids = new_ids

    return z

def smp():
    # Algorithm 5 in https://www.cs.utexas.edu/~ecprice/courses/sublinear/bwca-sparse-recovery.pdf
    pass

def ssmp():
    # Algorithm 1 here:
    # https://people.csail.mit.edu/indyk/ssmp.pdf
    pass


def simple(M, x, s: int, pinv=False):
    # First use transpose or least-squares to find a candidate subspace.
    # Then use least-squares to decode into it.
    # Similar to a single round of IHT, but IHT always decodes into the full
    # space, rather than a selection. So maybe it's more like a single round
    # of CoSaMP?
    k, n = M.shape
    if pinv:
        m1, _, _, _ = np.linalg.lstsq(M, x, rcond=None)
        ids = _support(m1, s)
    else:
        ids = _support(M.T @ x, s)
    m, _, _, _ = np.linalg.lstsq(M[:, ids], x, rcond=None)
    z = np.zeros(n)
    z[ids] = m
    return z

def simple2(M, x, s: int, pinv=False):
    # Like simple, but with an extra step: 2s -> s -> final
    k, n = M.shape
    if pinv:
        m1, _, _, _ = np.linalg.lstsq(M, x, rcond=None)
    else:
        m1 = M.T @ x
    ids = _support(m1, 2*s)

    m2, _, _, _ = np.linalg.lstsq(M[:, ids], x, rcond=None)
    ids = ids[_support(m2, s)]

    m3, _, _, _ = np.linalg.lstsq(M[:, ids], x, rcond=None)
    z = np.zeros(n)
    z[ids] = m3
    return z


def iht(M, x, s: int, max_iter=20, pinv=False):
    k, n = M.shape
    MPI = np.linalg.pinv(M) if pinv else M.T
    y = np.zeros(n)
    for it in range(max_iter):
        grad = MPI @ (M @ y - x)
        y = _treshold(y - grad, s)
    return y


def iht2(M, x, s: int, max_iter=20):
    k, n = M.shape
    y = np.zeros(n)
    for it in range(max_iter):
        g = M.T @ (M @ y - x)
        ids = _support(y, s)
        mu = (g[ids] ** 2).sum() / (((M[:, ids] @ g[ids]) ** 2).sum() + 1e-9)
        y = _treshold(y - mu * g, s)
    return y


def iht_ada(M, x, s: int, max_iter=20):
    # I can't really get this to work.
    k, n = M.shape
    y = np.zeros(n)
    g2 = 0
    for it in range(max_iter):
        g = M2.T @ (M2 @ y - x)
        g2 += g**2
        y = _treshold(y - g / np.sqrt(1e-9 + g2), s)
    return y

def make_count_sketch(batch_size, rows, columns):
    C = np.eye(rows)[np.random.randint(rows, size=(batch_size, columns))]
    C = C.transpose(0, 2, 1) # Put rows in the middle
    C *= 1 - 2.0 * np.random.randint(2, size=(batch_size, rows, columns))
    return C

def measure(method, d, sparsity, n_measures, reps=1000, distribution='gauss'):
    # Generate some measuring matrices
    #Gaussian:
    if distribution == 'gauss':
        Ms = np.random.randn(reps, n_measures, d)
    elif distribution == 'binary':
        Ms = 1 - 2. * np.random.randint(2, size=(reps, n_measures, d))
    elif distribution == 'count-sketch':
        Ms = make_count_sketch(reps, n_measures, d)
    elif distribution.endswith('count-sketch'):
        n = int(distribution[0])
        Ms = 0
        for _ in range(n):
            Ms += make_count_sketch(reps, n_measures, d)
    # All the methods assume column-norm is 1, so we just normalize here.
    Ms /= np.linalg.norm(Ms, axis=1, keepdims=True)
    # And some sparse vectors to measure.
    xs = np.zeros((reps, d))
    for x in xs:
        x[:sparsity] = 1 - 2 * np.random.randint(2, size=(sparsity,))
        #x[:sparsity] = np.random.randn(sparsity)
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
    n_sparsity = 20
    n_measurements = 50
    dim = 100
    reps = 10
    distribution = 'count-sketch'
    #distribution = 'gauss'
    #methods = [omp, cosamp2, cosamp, cosamp3, iht, iht2, partial(iht, pinv=True)]
    #titles = ["OMP", "CoSaMP", "CoSaMP1", "CoSaMP3", "IHT", "IHT 2", "IHT (pinv)"]
    # methods = [iht, iht2, partial(iht, pinv=True), cosamp]
    # titles = ['iht', 'iht2', 'iht (pinv)', 'cosamp']
    #methods = [omp, partial(omp, pinv=True), simple, partial(simple, pinv=True), simple2]
    #titles = ['omp', 'omp (pinv)', 'simple', 'simple (pinv)', 'simple2']
    methods = [omp, partial(omp,pinv=True), cosamp2, cosamp3, partial(cosamp2,pinv=True), partial(cosamp3,pinv=True), iht, iht2, partial(iht, pinv=True)]
    titles = ['omp', 'omp (pinv)', "CoSaMP", "CoSaMP3", "CoSaMP (pinv)", "CoSaMP3 (pinv)", 'iht', 'iht2', 'iht (pinv)']
    assert len(methods) == len(titles)
    fig, axs = plt.subplots(1, len(methods))
    fig.suptitle(f"Exact Recovery Rate with {dim} Dimensional Gaussian Measurements")
    for ax, method, title in zip(axs, methods, titles):
        data = np.zeros((n_measurements, n_sparsity))
        elapsed = 0
        for sp in tqdm.tqdm(range(n_sparsity)):
            for mes in tqdm.tqdm(range(n_measurements), leave=False):
                _err, recp, inner_elapsed = measure(
                    method, dim, sp + 1, mes + 1, reps=reps,
                    distribution=distribution
                )
                data[mes, sp] = recp
                elapsed += inner_elapsed
        ax.set_title(f"{title},\nt={elapsed:.1f}s,\np={data.mean()*100:.1f}%")
        print(title, data.mean())
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
