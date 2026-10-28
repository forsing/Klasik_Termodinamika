#!/usr/bin/env python3



"""
TERMODINAMIKA V1 — Loto 7/39

Canonical Gibbs distribution with exactly seven occupied states.

Dimensionless energy:

    E(y | H) =
        -sum_i h_i(H) * y_i
        -sum_i sum_d J_i,d(H) * y_i * y_(i-d)

    P(y | H) = exp(-E(y | H)) / Z(H)

    y_i in {0,1}
    sum_i y_i = 7

The learned coefficients represent beta*E.
Beta=1 fixes energy units and avoids an unidentifiable extra scale.

Learned local fields:
    static field;
    response to the previous draw;
    response to the draw before that.

Learned pair interactions:
    static interaction;
    response to that pair in the previous draw.

Interaction range 1 or 2 is selected by chronological validation.
Both attractive and repulsive interactions are allowed.
Every legal seven-number combination remains possible.

Exact computations:
    log partition function by transfer matrix;
    inclusion and pair marginals by forward/backward recursion;
    minimum energy by dynamic programming;
    mean energy, free energy and entropy.

No random sampling, shuffling or random splits.
No classifier.
No rounding or duplicate repair.
No cosmetic combination filters.
No quantum simulation.

Every available chronological transition enters final learning.
The first transition retains a neutral missing second-lag input.
"""



import os

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import csv
import hashlib
import math
import sys
import traceback
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


NUMBER_COUNT = 39
DRAW_SIZE = 7
HISTORY_LAGS = 2



FILES = (
    (
        Path("/data/loto7_4710_k86.csv"),
        4710,
    ),
    (
        Path("/data/loto7_4710_k86_loto_2977.csv"),
        2977,
    ),
    (
        Path("/data/loto7_4710_k86_loto_plus_1733.csv"),
        1733,
    ),
)



CANDIDATES = tuple(
    (interaction_range, ridge)
    for interaction_range in (1, 2)
    for ridge in (0.03, 0.3, 3.0)
)


def say(*lines):
    for line in lines:
        print(line, flush=True)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def edge_mask(number_count, interaction_range):
    return (
        np.arange(number_count)[:, None]
        > np.arange(interaction_range)[None, :]
    ).astype(np.float64)


def pair_features(occupancy, interaction_range):
    sample_count, number_count = occupancy.shape

    result = np.zeros(
        (sample_count, number_count, interaction_range),
        dtype=np.float64,
    )

    for index in range(interaction_range):
        distance = index + 1

        result[:, distance:, index] = (
            occupancy[:, distance:]
            * occupancy[:, :-distance]
        )

    return result


def history_context(occupancy, target_indices, draw_size):
    target_indices = np.asarray(
        target_indices,
        dtype=np.int64,
    )

    require(
        target_indices.ndim == 1
        and np.all(target_indices >= 1)
        and np.all(target_indices <= len(occupancy)),
        "Neispravni ciljni indeksi.",
    )

    number_count = occupancy.shape[1]

    result = np.zeros(
        (
            len(target_indices),
            HISTORY_LAGS,
            number_count,
        ),
        dtype=np.float64,
    )

    # Exactly k/m is the mean occupancy across labels in any legal draw.
    # This centers inputs; it does not impose an output distribution.
    density = draw_size / number_count

    for lag in range(1, HISTORY_LAGS + 1):
        previous = target_indices - lag
        valid = np.flatnonzero(previous >= 0)

        if valid.size:
            result[valid, lag - 1, :] = (
                occupancy[previous[valid]] - density
            )

    return result


def make_data(occupancy, draw_size, interaction_range):
    targets = np.arange(
        1,
        len(occupancy),
        dtype=np.int64,
    )

    return {
        "x": history_context(
            occupancy,
            targets,
            draw_size,
        ),
        "hp": pair_features(
            occupancy[:-1],
            interaction_range,
        ),
        "y": occupancy[1:].astype(np.float64),
        "yp": pair_features(
            occupancy[1:],
            interaction_range,
        ),
    }


def subset(data, start, stop):
    return {
        key: value[start:stop]
        for key, value in data.items()
    }


def parameter_count(number_count, interaction_range):
    return (
        (HISTORY_LAGS + 1) * number_count
        + 2 * number_count * interaction_range
    )


def unpack(parameters, number_count, interaction_range):
    field_size = (
        (HISTORY_LAGS + 1) * number_count
    )
    pair_size = (
        number_count * interaction_range
    )

    field_parameters = parameters[:field_size].reshape(
        HISTORY_LAGS + 1,
        number_count,
    )

    static_pairs = parameters[
        field_size:field_size + pair_size
    ].reshape(
        number_count,
        interaction_range,
    )

    history_pairs = parameters[
        field_size + pair_size:
    ].reshape(
        number_count,
        interaction_range,
    )

    return (
        field_parameters,
        static_pairs,
        history_pairs,
    )


def potentials(parameters, context, historical_pairs, interaction_range):
    number_count = context.shape[2]

    fields, static_pairs, history_pairs = unpack(
        parameters,
        number_count,
        interaction_range,
    )

    local_fields = (
        fields[0][None, :]
        + np.einsum(
            "nlm,lm->nm",
            context,
            fields[1:],
            optimize=True,
        )
    )

    couplings = (
        static_pairs[None, :, :]
        + historical_pairs * history_pairs[None, :, :]
    )

    couplings *= edge_mask(
        number_count,
        interaction_range,
    )[None, :, :]

    require(
        np.all(np.isfinite(local_fields))
        and np.all(np.isfinite(couplings)),
        "Energetski parametri nisu konačni.",
    )

    return local_fields, couplings


def infer(fields, couplings, draw_size, marginals=False):
    """
    Exact fixed-particle canonical partition function.

    State bits retain the most recently selected positions.
    State bit 0 refers to the previous position.
    """
    fields = np.asarray(fields, dtype=np.float64)
    couplings = np.asarray(couplings, dtype=np.float64)

    require(
        fields.ndim == 2
        and couplings.ndim == 3
        and fields.shape == couplings.shape[:2],
        "Neispravan oblik Hamiltonijana.",
    )

    sample_count, number_count = fields.shape
    interaction_range = couplings.shape[2]

    require(
        sample_count > 0
        and 1 <= draw_size <= number_count
        and interaction_range in (1, 2),
        "Neispravni parametri transfer-matrice.",
    )

    require(
        np.all(np.isfinite(fields))
        and np.all(np.isfinite(couplings)),
        "Hamiltonijan sadrži neispravne vrednosti.",
    )

    state_count = 1 << interaction_range
    state_mask = state_count - 1

    bits = np.array(
        [
            [
                (state >> distance) & 1
                for distance in range(interaction_range)
            ]
            for state in range(state_count)
        ],
        dtype=np.float64,
    )

    valid_edges = edge_mask(
        number_count,
        interaction_range,
    )

    forward = np.full(
        (
            sample_count,
            draw_size + 1,
            state_count,
        ),
        -np.inf,
        dtype=np.float64,
    )
    forward[:, 0, 0] = 0.0

    cache = [forward] if marginals else None

    for number in range(number_count):
        local_score = (
            fields[:, number, None]
            + (
                couplings[:, number, :]
                * valid_edges[number]
            ) @ bits.T
        )

        following = np.full_like(
            forward,
            -np.inf,
        )

        for state in range(state_count):
            destination_zero = (
                state << 1
            ) & state_mask

            destination_one = (
                destination_zero | 1
            )

            following[:, :, destination_zero] = np.logaddexp(
                following[:, :, destination_zero],
                forward[:, :, state],
            )

            following[:, 1:, destination_one] = np.logaddexp(
                following[:, 1:, destination_one],
                forward[:, :-1, state]
                + local_score[:, state, None],
            )

        forward = following

        if marginals:
            cache.append(forward)

    log_z = logsumexp(
        forward[:, draw_size, :],
        axis=1,
    )

    require(
        np.all(np.isfinite(log_z)),
        "Particiona funkcija nije konačna.",
    )

    if not marginals:
        return log_z, None, None

    backward = np.full_like(
        forward,
        -np.inf,
    )
    backward[:, draw_size, :] = 0.0

    inclusion = np.zeros_like(fields)
    pair_marginals = np.zeros_like(couplings)

    for number in range(number_count - 1, -1, -1):
        previous = cache[number]

        local_score = (
            fields[:, number, None]
            + (
                couplings[:, number, :]
                * valid_edges[number]
            ) @ bits.T
        )

        previous_backward = np.full_like(
            backward,
            -np.inf,
        )

        for state in range(state_count):
            destination_zero = (
                state << 1
            ) & state_mask

            destination_one = (
                destination_zero | 1
            )

            select_suffix = np.full(
                (sample_count, draw_size + 1),
                -np.inf,
            )

            select_suffix[:, :-1] = (
                local_score[:, state, None]
                + backward[:, 1:, destination_one]
            )

            previous_backward[:, :, state] = np.logaddexp(
                backward[:, :, destination_zero],
                select_suffix,
            )

            log_edge_probability = (
                previous[:, :-1, state]
                + local_score[:, state, None]
                + backward[:, 1:, destination_one]
                - log_z[:, None]
            )

            selected_probability = np.sum(
                np.exp(log_edge_probability),
                axis=1,
            )

            inclusion[:, number] += (
                selected_probability
            )

            pair_marginals[:, number, :] += (
                selected_probability[:, None]
                * bits[state][None, :]
                * valid_edges[number][None, :]
            )

        backward = previous_backward

    require(
        np.max(
            np.abs(
                backward[:, 0, 0] - log_z
            )
        ) < 1e-8,
        "Prednja i zadnja particiona funkcija nisu saglasne.",
    )

    require(
        np.all(np.isfinite(inclusion))
        and np.all(np.isfinite(pair_marginals)),
        "Marginalne verovatnoće nisu konačne.",
    )

    require(
        np.min(inclusion) >= -1e-9
        and np.max(inclusion) <= 1.0 + 1e-9,
        "Uključne verovatnoće su van opsega [0,1].",
    )

    require(
        np.max(
            np.abs(
                inclusion.sum(axis=1) - draw_size
            )
        ) < 1e-7,
        "Zbir uključnih verovatnoća nije sedam.",
    )

    return (
        log_z,
        inclusion,
        pair_marginals,
    )


def observed_score(fields, couplings, occupancy, pairs):
    return (
        np.sum(fields * occupancy, axis=1)
        + np.sum(
            couplings * pairs,
            axis=(1, 2),
        )
    )


def objective(parameters, data, interaction_range, ridge, draw_size):
    number_count = data["y"].shape[1]

    fields, couplings = potentials(
        parameters,
        data["x"],
        data["hp"],
        interaction_range,
    )

    log_z, inclusion, pair_marginals = infer(
        fields,
        couplings,
        draw_size,
        marginals=True,
    )

    scores = observed_score(
        fields,
        couplings,
        data["y"],
        data["yp"],
    )

    nll = log_z - scores

    require(
        np.all(np.isfinite(nll))
        and np.min(nll) >= -1e-7,
        "Neispravna verovatnoća posmatrane kombinacije.",
    )

    residual_fields = (
        inclusion - data["y"]
    )
    residual_pairs = (
        pair_marginals - data["yp"]
    )

    sample_count = len(nll)

    gradient_fields = np.empty(
        (HISTORY_LAGS + 1, number_count),
        dtype=np.float64,
    )

    gradient_fields[0] = (
        residual_fields.mean(axis=0)
    )

    gradient_fields[1:] = (
        np.einsum(
            "nm,nlm->lm",
            residual_fields,
            data["x"],
            optimize=True,
        ) / sample_count
    )

    mask = edge_mask(
        number_count,
        interaction_range,
    )

    gradient_static_pairs = (
        residual_pairs.mean(axis=0) * mask
    )

    gradient_history_pairs = (
        np.mean(
            residual_pairs * data["hp"],
            axis=0,
        ) * mask
    )

    gradient = np.concatenate(
        (
            gradient_fields.ravel(),
            gradient_static_pairs.ravel(),
            gradient_history_pairs.ravel(),
        )
    )

    penalty_weights = np.ones_like(parameters)
    penalty_weights[:number_count] = 0.1

    penalty = ridge * np.sum(
        penalty_weights * parameters * parameters
    )

    gradient += (
        2.0
        * ridge
        * penalty_weights
        * parameters
    )

    value = float(
        np.mean(nll) + penalty
    )

    require(
        np.isfinite(value)
        and np.all(np.isfinite(gradient)),
        "Funkcija gubitka ili njen izvod nisu konačni.",
    )

    return value, gradient


def minimum_energy(fields, couplings, draw_size):
    """
    Exact maximum of -E over all legal subsets.

    Dynamic programming retains:
        selected count;
        finite-range state;
        lexicographically first maximizing path.
    """
    fields = np.asarray(fields, dtype=np.float64)
    couplings = np.asarray(couplings, dtype=np.float64)

    number_count = len(fields)
    interaction_range = couplings.shape[1]

    state_count = 1 << interaction_range
    state_mask = state_count - 1

    valid_edges = edge_mask(
        number_count,
        interaction_range,
    )

    values = np.full(
        (draw_size + 1, state_count),
        -np.inf,
    )

    paths = [
        [None] * state_count
        for _ in range(draw_size + 1)
    ]

    values[0, 0] = 0.0
    paths[0][0] = ()

    for number in range(number_count):
        following = np.full_like(
            values,
            -np.inf,
        )
        following_paths = [
            [None] * state_count
            for _ in range(draw_size + 1)
        ]

        for count in range(draw_size + 1):
            for state in range(state_count):
                path = paths[count][state]

                if path is None:
                    continue

                interaction_score = sum(
                    couplings[number, distance]
                    * valid_edges[number, distance]
                    * ((state >> distance) & 1)
                    for distance in range(interaction_range)
                )

                for selected in (0, 1):
                    new_count = count + selected

                    if new_count > draw_size:
                        continue

                    destination = (
                        (state << 1) | selected
                    ) & state_mask

                    candidate_score = (
                        values[count, state]
                        + selected
                        * (
                            fields[number]
                            + interaction_score
                        )
                    )

                    candidate_path = (
                        path + (number,)
                        if selected
                        else path
                    )

                    old_path = following_paths[
                        new_count
                    ][destination]

                    old_score = following[
                        new_count,
                        destination,
                    ]

                    if (
                        old_path is None
                        or candidate_score > old_score
                        or (
                            candidate_score == old_score
                            and candidate_path < old_path
                        )
                    ):
                        following[
                            new_count,
                            destination,
                        ] = candidate_score

                        following_paths[
                            new_count
                        ][destination] = candidate_path

        values = following
        paths = following_paths

    candidates = [
        (
            float(values[draw_size, state]),
            paths[draw_size][state],
        )
        for state in range(state_count)
        if paths[draw_size][state] is not None
    ]

    require(
        bool(candidates),
        "Nije pronađeno dozvoljeno stanje najmanje energije.",
    )

    best_score = max(
        value for value, _ in candidates
    )

    best_path = min(
        path
        for value, path in candidates
        if value == best_score
    )

    require(
        len(best_path) == draw_size
        and len(set(best_path)) == draw_size,
        "Stanje najmanje energije nema ispravnu veličinu.",
    )

    return best_path, best_score


def fit(data, interaction_range, ridge, draw_size, label):
    number_count = data["y"].shape[1]

    parameters = np.zeros(
        parameter_count(
            number_count,
            interaction_range,
        ),
        dtype=np.float64,
    )

    iterations = 0

    def callback(current):
        nonlocal iterations
        iterations += 1

        if iterations % 25 == 0:
            say(
                f"  {label}: {iterations} iteracija"
            )

    result = None
    gradient_norm = math.inf

    for _ in range(3):
        result = minimize(
            objective,
            parameters,
            args=(
                data,
                interaction_range,
                ridge,
                draw_size,
            ),
            method="L-BFGS-B",
            jac=True,
            callback=callback,
            options={
                "maxiter": 500,
                "maxls": 60,
                "maxcor": 20,
                "ftol": 1e-12,
                "gtol": 1e-6,
            },
        )

        parameters = result.x.copy()

        value, gradient = objective(
            parameters,
            data,
            interaction_range,
            ridge,
            draw_size,
        )

        gradient_norm = float(
            np.max(np.abs(gradient))
        )

        if (
            result.success
            and gradient_norm <= 5e-5
        ):
            break

    require(
        result.success
        and gradient_norm <= 5e-5,
        f"{label}: optimizacija nije prošla; "
        f"{result.message}; max|grad|={gradient_norm:.3e}.",
    )

    say(
        f"{label}: R={interaction_range}, lambda={ridge}, "
        f"gubitak={value:.8f}, "
        f"max|grad|={gradient_norm:.3e}"
    )

    return parameters


def evaluate(parameters, data, interaction_range, draw_size, hits=False):
    total = 0.0

    histogram = np.zeros(
        draw_size + 1,
        dtype=np.int64,
    )

    for start in range(0, len(data["y"]), 512):
        stop = min(
            start + 512,
            len(data["y"]),
        )

        block = subset(data, start, stop)

        fields, couplings = potentials(
            parameters,
            block["x"],
            block["hp"],
            interaction_range,
        )

        log_z = infer(
            fields,
            couplings,
            draw_size,
        )[0]

        scores = observed_score(
            fields,
            couplings,
            block["y"],
            block["yp"],
        )

        nll = log_z - scores

        require(
            np.all(np.isfinite(nll))
            and np.min(nll) >= -1e-7,
            "Neispravan evaluacioni NLL.",
        )

        total += float(nll.sum())

        if hits:
            for row in range(len(fields)):
                prediction, _ = minimum_energy(
                    fields[row],
                    couplings[row],
                    draw_size,
                )

                actual = set(
                    np.flatnonzero(block["y"][row]).tolist()
                )

                matched = len(
                    set(prediction).intersection(actual)
                )

                histogram[matched] += 1

    return (
        total / len(data["y"]),
        histogram,
    )


def load_draws(path, expected_rows):
    fingerprint = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    rows = []
    blank_rows = 0

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for line_number, row in enumerate(
            csv.reader(handle),
            start=1,
        ):
            if not row or all(
                not item.strip() for item in row
            ):
                blank_rows += 1
                continue

            require(
                len(row) == DRAW_SIZE,
                f"{path.name}, red {line_number}: potrebno je 7 kolona.",
            )

            try:
                values = [
                    int(item.strip()) for item in row
                ]
            except ValueError as error:
                raise RuntimeError(
                    f"{path.name}, red {line_number}: neispravan ceo broj."
                ) from error

            require(
                len(set(values)) == DRAW_SIZE,
                f"{path.name}, red {line_number}: ponovljen broj.",
            )

            require(
                all(
                    1 <= value <= NUMBER_COUNT
                    for value in values
                ),
                f"{path.name}, red {line_number}: broj van opsega 1–39.",
            )

            rows.append(
                sorted(value - 1 for value in values)
            )

    require(
        len(rows) == expected_rows,
        f"{path.name}: očekivano {expected_rows}, pročitano {len(rows)}.",
    )

    draws = np.asarray(
        rows,
        dtype=np.int64,
    )

    occupancy = np.zeros(
        (len(draws), NUMBER_COUNT),
        dtype=np.float64,
    )
    occupancy[
        np.arange(len(draws))[:, None],
        draws,
    ] = 1.0

    say(
        "",
        f"ULAZ: {path.name}",
        f"Redova: {len(draws)} | praznih redova: {blank_rows}",
        f"SHA256: {fingerprint}",
        "Redosled kola ostaje redosled iz CSV-a.",
    )

    return draws, occupancy, fingerprint


def self_checks():
    say(
        "",
        "Pokrećem matematičke i numeričke samoprovere."
    )

    toy_draws = np.array(
        [
            [0, 2, 4],
            [1, 3, 6],
            [0, 5, 6],
            [2, 3, 5],
        ],
        dtype=np.int64,
    )

    number_count = 7
    draw_size = 3
    interaction_range = 2

    occupancy = np.zeros(
        (len(toy_draws), number_count),
        dtype=np.float64,
    )
    occupancy[
        np.arange(len(toy_draws))[:, None],
        toy_draws,
    ] = 1.0

    data = make_data(
        occupancy,
        draw_size,
        interaction_range,
    )

    parameters = (
        0.04
        * np.sin(
            np.arange(
                parameter_count(
                    number_count,
                    interaction_range,
                )
            ) + 0.3
        )
    )

    _, analytic = objective(
        parameters,
        data,
        interaction_range,
        0.03,
        draw_size,
    )

    numerical = np.empty_like(parameters)

    for index in range(len(parameters)):
        plus = parameters.copy()
        minus = parameters.copy()

        plus[index] += 1e-6
        minus[index] -= 1e-6

        numerical[index] = (
            objective(
                plus,
                data,
                interaction_range,
                0.03,
                draw_size,
            )[0]
            - objective(
                minus,
                data,
                interaction_range,
                0.03,
                draw_size,
            )[0]
        ) / 2e-6

    gradient_error = float(
        np.max(
            np.abs(analytic - numerical)
            / (
                1.0
                + np.abs(analytic)
                + np.abs(numerical)
            )
        )
    )

    require(
        gradient_error < 2e-5,
        "Provera izvoda funkcije gubitka nije prošla.",
    )

    fields, couplings = potentials(
        parameters,
        data["x"],
        data["hp"],
        interaction_range,
    )

    log_z, inclusion, pair_marginals = infer(
        fields,
        couplings,
        draw_size,
        marginals=True,
    )

    all_sets = list(
        combinations(
            range(number_count),
            draw_size,
        )
    )

    all_occupancy = np.zeros(
        (len(all_sets), number_count),
        dtype=np.float64,
    )

    for row, selected in enumerate(all_sets):
        all_occupancy[
            row,
            np.asarray(selected, dtype=np.int64),
        ] = 1.0

    all_pairs = pair_features(
        all_occupancy,
        interaction_range,
    )

    for row in range(len(fields)):
        scores = (
            all_occupancy @ fields[row]
            + np.sum(
                all_pairs * couplings[row][None, :, :],
                axis=(1, 2),
            )
        )

        direct_log_z = float(
            logsumexp(scores)
        )

        probabilities = np.exp(
            scores - direct_log_z
        )

        require(
            abs(direct_log_z - log_z[row]) < 1e-10,
            "Particiona funkcija nije prošla potpunu proveru.",
        )

        require(
            abs(float(probabilities.sum()) - 1.0) < 1e-12,
            "Mala Gibbsova raspodela nije normalizovana.",
        )

        direct_inclusion = (
            probabilities @ all_occupancy
        )

        direct_pairs = np.einsum(
            "n,nmr->mr",
            probabilities,
            all_pairs,
        )

        require(
            np.max(
                np.abs(
                    direct_inclusion - inclusion[row]
                )
            ) < 1e-10,
            "Uključne verovatnoće nisu prošle proveru.",
        )

        require(
            np.max(
                np.abs(
                    direct_pairs - pair_marginals[row]
                )
            ) < 1e-10,
            "Verovatnoće parova nisu prošle proveru.",
        )

        best, best_score = minimum_energy(
            fields[row],
            couplings[row],
            draw_size,
        )

        require(
            abs(best_score - float(scores.max())) < 1e-10
            and abs(
                best_score
                - float(scores[all_sets.index(best)])
            ) < 1e-10,
            "Stanje najmanje energije nije prošlo proveru.",
        )

        mean_score = float(
            probabilities @ scores
        )

        entropy = (
            direct_log_z - mean_score
        )

        direct_entropy = float(
            -np.sum(
                probabilities * np.log(probabilities)
            )
        )

        require(
            abs(entropy - direct_entropy) < 1e-10,
            "Termodinamička entropija nije prošla proveru.",
        )

        epsilon = 1e-5

        log_plus = infer(
            fields[row:row + 1] * (1 + epsilon),
            couplings[row:row + 1] * (1 + epsilon),
            draw_size,
        )[0][0]

        log_minus = infer(
            fields[row:row + 1] * (1 - epsilon),
            couplings[row:row + 1] * (1 - epsilon),
            draw_size,
        )[0][0]

        derivative = (
            log_plus - log_minus
        ) / (2 * epsilon)

        require(
            abs(derivative - mean_score) < 1e-7,
            "Izvod particione funkcije po beta nije prošao proveru.",
        )

    for model_range in (1, 2):
        for selected_count in (1, number_count):
            zero_fields = np.zeros(
                (1, number_count)
            )
            zero_pairs = np.zeros(
                (1, number_count, model_range)
            )

            z, marginal, _ = infer(
                zero_fields,
                zero_pairs,
                selected_count,
                marginals=True,
            )

            require(
                abs(
                    z[0]
                    - math.log(
                        math.comb(
                            number_count,
                            selected_count,
                        )
                    )
                ) < 1e-10,
                "Granična provera particione funkcije nije prošla.",
            )

            require(
                np.max(
                    np.abs(
                        marginal
                        - selected_count / number_count
                    )
                ) < 1e-10,
                "Granična provera marginala nije prošla.",
            )

            best, _ = minimum_energy(
                zero_fields[0],
                zero_pairs[0],
                selected_count,
            )

            require(
                best == tuple(range(selected_count)),
                "Deterministički izbor pri izjednačenju nije ispravan.",
            )

    require(
        np.all(data["x"][0, 1] == 0)
        and np.all(data["y"][0] == occupancy[1]),
        "Početni istorijski prelaz nije ispravan.",
    )

    say(
        "Samoprovere prošle: energija, izvodi, normalizacija,",
        "marginale, parovi, entropija, beta izvod i tačan maksimum.",
        f"Relativna greška izvoda: {gradient_error:.3e}",
    )


def longest_run(combination):
    longest = 1
    current = 1

    for left, right in zip(
        combination,
        combination[1:],
    ):
        if right == left + 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1

    return longest


def process_file(path, draws, occupancy, fingerprint):
    sample_count = len(draws) - 1

    train_60 = int(0.60 * sample_count)
    train_70 = int(0.70 * sample_count)
    train_80 = int(0.80 * sample_count)

    require(
        50 < train_60 < train_70 < train_80 < sample_count,
        "Nema dovoljno podataka za hronološku podelu.",
    )

    data_cache = {
        model_range: make_data(
            occupancy,
            DRAW_SIZE,
            model_range,
        )
        for model_range in (1, 2)
    }

    scores = []

    for model_range, ridge in CANDIDATES:
        data = data_cache[model_range]

        total_nll = 0.0
        total_count = 0

        folds = (
            (train_60, train_70),
            (train_70, train_80),
        )

        for fold_number, (end, stop) in enumerate(
            folds,
            start=1,
        ):
            parameters = fit(
                subset(data, 0, end),
                model_range,
                ridge,
                DRAW_SIZE,
                f"{path.name} / validacija {fold_number}",
            )

            nll = evaluate(
                parameters,
                subset(data, end, stop),
                model_range,
                DRAW_SIZE,
            )[0]

            total_nll += (
                (stop - end) * nll
            )
            total_count += stop - end

            say(
                f"  Validacioni NLL: {nll:.8f}"
            )

        score = total_nll / total_count

        scores.append(
            (score, model_range, ridge)
        )

        say(
            f"KANDIDAT: R={model_range}, lambda={ridge}, "
            f"validacioni NLL={score:.8f}"
        )

    validation_nll, model_range, ridge = min(
        scores,
        key=lambda item: item[0],
    )

    data = data_cache[model_range]

    say(
        "",
        f"IZABRANO: R={model_range}, lambda={ridge}, "
        f"validacioni NLL={validation_nll:.8f}",
    )

    test_parameters = fit(
        subset(data, 0, train_80),
        model_range,
        ridge,
        DRAW_SIZE,
        f"{path.name} / odvojeni test",
    )

    test_nll, histogram = evaluate(
        test_parameters,
        subset(data, train_80, sample_count),
        model_range,
        DRAW_SIZE,
        hits=True,
    )

    say(
        f"ODVOJENI TEST: NLL={test_nll:.8f}, "
        f"kola={sample_count - train_80}",
        "Pogodaka 0–7: "
        + " ".join(
            str(int(value))
            for value in histogram
        ),
        "Test se ne koristi za promenu parametara.",
    )

    final_parameters = fit(
        data,
        model_range,
        ridge,
        DRAW_SIZE,
        f"{path.name} / svih {sample_count} prelaza",
    )

    next_context = history_context(
        occupancy,
        [len(occupancy)],
        DRAW_SIZE,
    )

    next_history_pairs = pair_features(
        occupancy[-1:],
        model_range,
    )

    fields, couplings = potentials(
        final_parameters,
        next_context,
        next_history_pairs,
        model_range,
    )

    log_z, marginal, pair_marginal = infer(
        fields,
        couplings,
        DRAW_SIZE,
        marginals=True,
    )

    best, best_score = minimum_energy(
        fields[0],
        couplings[0],
        DRAW_SIZE,
    )

    log_probability = (
        best_score - float(log_z[0])
    )

    require(
        log_probability <= 1e-7
        and log_probability >= (
            -math.log(
                math.comb(
                    NUMBER_COUNT,
                    DRAW_SIZE,
                )
            ) - 1e-7
        ),
        "Verovatnoća maksimuma nije saglasna sa normalizacijom.",
    )

    mean_score = float(
        np.sum(fields * marginal)
        + np.sum(couplings * pair_marginal)
    )

    mean_energy = -mean_score
    free_energy = -float(log_z[0])
    entropy = float(log_z[0]) - mean_score

    require(
        -1e-7 <= entropy <= (
            math.log(
                math.comb(
                    NUMBER_COUNT,
                    DRAW_SIZE,
                )
            ) + 1e-7
        ),
        "Entropija je van dozvoljenih granica.",
    )

    previous_fields, previous_couplings = potentials(
        final_parameters,
        data["x"][-8:],
        data["hp"][-8:],
        model_range,
    )

    previous_marginals = infer(
        previous_fields,
        previous_couplings,
        DRAW_SIZE,
        marginals=True,
    )[1]

    sensitivity = float(
        np.max(
            np.abs(
                previous_marginals - marginal
            )
        )
    )

    repetitions = int(
        np.count_nonzero(
            np.all(
                draws == np.asarray(best)[None, :],
                axis=1,
            )
        )
    )

    run_length = longest_run(best)
    span = best[-1] - best[0]

    warnings_list = []

    if run_length >= 5:
        warnings_list.append(
            f"Niz od {run_length} uzastopnih brojeva "
            "zahteva pregled modela."
        )

    if span <= 9:
        warnings_list.append(
            "Svih sedam brojeva nalazi se unutar "
            "deset uzastopnih vrednosti. Proveriti grupisanje."
        )

    if repetitions:
        warnings_list.append(
            f"NEXT se pojavljuje {repetitions} puta u istoriji."
        )

    if sensitivity < 1e-8:
        warnings_list.append(
            "Na proverenim kontekstima nije utvrđen "
            "numerički uticaj promene istorijskog ulaza."
        )

    test_count = sample_count - train_80

    return {
        "name": path.name,
        "fingerprint": fingerprint,
        "next": tuple(value + 1 for value in best),
        "range": model_range,
        "ridge": ridge,
        "validation_nll": validation_nll,
        "test_nll": test_nll,
        "histogram": histogram,
        "mean_hits": float(
            histogram @ np.arange(DRAW_SIZE + 1)
            / test_count
        ),
        "probability": math.exp(log_probability),
        "log_z": float(log_z[0]),
        "minimum_energy": -best_score,
        "mean_energy": mean_energy,
        "free_energy": free_energy,
        "entropy": entropy,
        "marginal_sum": float(marginal.sum()),
        "sensitivity": sensitivity,
        "warnings": warnings_list,
    }


def main():
    np.seterr(
        over="raise",
        invalid="raise",
        divide="raise",
        under="ignore",
    )

    say(
        "TERMODINAMIKA V1 — Loto 7/39",
        "Naučeni Hamiltonijan i kanonska Gibbsova raspodela.",
        "Energije brojeva, interakcije parova i istorijski prelazi.",
        "Tačna transfer-matrica i stanje najmanje energije.",
        "Beta=1; energija je izražena u bezdimenzionim jedinicama.",
        "Ulaz uzorka: prethodna dva kola.",
        "Završno učenje: svi raspoloživi istorijski prelazi.",
        "Bez slučajnog uzorkovanja i bez klasifikatora.",
    )

    datasets = []

    for path, expected_rows in FILES:
        draws, occupancy, fingerprint = load_draws(
            path,
            expected_rows,
        )

        datasets.append(
            (
                path,
                draws,
                occupancy,
                fingerprint,
            )
        )

    require(
        len(
            {
                path.resolve()
                for path, _, _, _ in datasets
            }
        ) == 3,
        "Putanje CSV-ova nisu različite.",
    )

    require(
        len(
            {
                fingerprint
                for _, _, _, fingerprint in datasets
            }
        ) == 3,
        "Dva CSV-a imaju identičan sadržaj.",
    )

    self_checks()

    results = [
        process_file(*dataset)
        for dataset in datasets
    ]

    for left in range(len(results)):
        for right in range(left + 1, len(results)):
            if (
                results[left]["next"]
                == results[right]["next"]
            ):
                message = (
                    "Ista NEXT kombinacija za "
                    f"{results[left]['name']} i "
                    f"{results[right]['name']}. "
                    "Proveriti istorijski ulaz i degeneraciju. "
                    "Rezultat nije promenjen."
                )

                results[left]["warnings"].append(message)
                results[right]["warnings"].append(message)

    say(
        "",
        "ZAVRŠNI REZULTATI TERMODINAMIKA V1",
        "Energija, slobodna energija i entropija "
        "pripadaju matematičkom modelu.",
        "Modelna verovatnoća nije potvrđena "
        "verovatnoća stvarnog izvlačenja.",
    )

    for result in results:
        say(
            "",
            f"CSV: {result['name']}",
            "NEXT: "
            + " ".join(
                f"{value:02d}"
                for value in result["next"]
            ),
            f"Izabrani domet interakcije: {result['range']}",
            f"Izabrana regularizacija: {result['ridge']}",
            f"Validacioni NLL: {result['validation_nll']:.8f}",
            f"Odvojeni test NLL: {result['test_nll']:.8f}",
            "Prosečno pogodaka na odvojenom testu: "
            f"{result['mean_hits']:.6f}",
            "Test pogodaka 0–7: "
            + " ".join(
                str(int(value))
                for value in result["histogram"]
            ),
            "Modelna verovatnoća NEXT: "
            f"{result['probability']:.12g}",
            f"log Z: {result['log_z']:.10f}",
            f"Najmanja energija NEXT: {result['minimum_energy']:.10f}",
            f"Srednja energija U: {result['mean_energy']:.10f}",
            f"Slobodna energija F: {result['free_energy']:.10f}",
            f"Entropija S: {result['entropy']:.10f}",
            "Zbir uključnih verovatnoća: "
            f"{result['marginal_sum']:.12f}",
            "Promena uključnih verovatnoća kroz kontekst: "
            f"{result['sensitivity']:.6e}",
        )

        for message in result["warnings"]:
            say(
                "PROVERA MODELA: " + message
            )

    say(
        "",
        "NEXT je tačno stanje najmanje naučene energije.",
        "Direktne interakcije postoje samo unutar "
        "izabranog dometa 1 ili 2.",
        "Prediktivna prednost zahteva procenu "
        "prikazanih rezultata odvojenog testa.",
    )


if __name__ == "__main__":
    try:
        main()

    except (
        OSError,
        UnicodeError,
        csv.Error,
        ValueError,
        RuntimeError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as error:
        print(
            f"ZAUSTAVLJENO: {error}",
            file=sys.stderr,
            flush=True,
        )
        print(
            "Završni skup od tri NEXT rezultata nije potvrđen.",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        sys.exit(1)



"""
TERMODINAMIKA V1 — Loto 7/39
Naučeni Hamiltonijan i kanonska Gibbsova raspodela.
Energije brojeva, interakcije parova i istorijski prelazi.
Tačna transfer-matrica i stanje najmanje energije.
Beta=1; energija je izražena u bezdimenzionim jedinicama.
Ulaz uzorka: prethodna dva kola.
Završno učenje: svi raspoloživi istorijski prelazi.
Bez slučajnog uzorkovanja i bez klasifikatora.

ULAZ: loto7_4710_k86.csv
Redova: 4710 | praznih redova: 0
SHA256: e51a5ab684401459535996f50fd0b0540b0af6d2b86bca49825cf381d759c85d
Redosled kola ostaje redosled iz CSV-a.

ULAZ: loto7_4710_k86_loto_2977.csv
Redova: 2977 | praznih redova: 0
SHA256: 69ae55edbff87383f9bb4a25df4ba6b44acbf23404db6e7fa3642de0442479f4
Redosled kola ostaje redosled iz CSV-a.

ULAZ: loto7_4710_k86_loto_plus_1733.csv
Redova: 1733 | praznih redova: 0
SHA256: 9d708eb88434b55a38bbc69474a869955e0295f1bed401cf10dc177d7d8fef39
Redosled kola ostaje redosled iz CSV-a.

Pokrećem matematičke i numeričke samoprovere.
Samoprovere prošle: energija, izvodi, normalizacija,
marginale, parovi, entropija, beta izvod i tačan maksimum.
Relativna greška izvoda: 3.261e-10
loto7_4710_k86.csv / validacija 1: R=1, lambda=0.03, gubitak=16.53697651, max|grad|=4.547e-07
  Validacioni NLL: 16.55432593
loto7_4710_k86.csv / validacija 2: R=1, lambda=0.03, gubitak=16.53861823, max|grad|=4.856e-07
  Validacioni NLL: 16.55373478
KANDIDAT: R=1, lambda=0.03, validacioni NLL=16.55403036
loto7_4710_k86.csv / validacija 1: R=1, lambda=0.3, gubitak=16.54339394, max|grad|=8.594e-07
  Validacioni NLL: 16.55103533
loto7_4710_k86.csv / validacija 2: R=1, lambda=0.3, gubitak=16.54411558, max|grad|=4.443e-07
  Validacioni NLL: 16.54911882
KANDIDAT: R=1, lambda=0.3, validacioni NLL=16.55007708
loto7_4710_k86.csv / validacija 1: R=1, lambda=3.0, gubitak=16.54729036, max|grad|=2.252e-08
  Validacioni NLL: 16.54866957
loto7_4710_k86.csv / validacija 2: R=1, lambda=3.0, gubitak=16.54747550, max|grad|=1.517e-08
  Validacioni NLL: 16.54813076
KANDIDAT: R=1, lambda=3.0, validacioni NLL=16.54840017
loto7_4710_k86.csv / validacija 1: R=2, lambda=0.03, gubitak=16.53573970, max|grad|=4.092e-07
  Validacioni NLL: 16.55549480
loto7_4710_k86.csv / validacija 2: R=2, lambda=0.03, gubitak=16.53756603, max|grad|=5.390e-07
  Validacioni NLL: 16.55355024
KANDIDAT: R=2, lambda=0.03, validacioni NLL=16.55452252
loto7_4710_k86.csv / validacija 1: R=2, lambda=0.3, gubitak=16.54321553, max|grad|=8.502e-07
  Validacioni NLL: 16.55110195
loto7_4710_k86.csv / validacija 2: R=2, lambda=0.3, gubitak=16.54395715, max|grad|=2.430e-08
  Validacioni NLL: 16.54910835
KANDIDAT: R=2, lambda=0.3, validacioni NLL=16.55010515
loto7_4710_k86.csv / validacija 1: R=2, lambda=3.0, gubitak=16.54726502, max|grad|=1.772e-06
  Validacioni NLL: 16.54865775
loto7_4710_k86.csv / validacija 2: R=2, lambda=3.0, gubitak=16.54745123, max|grad|=1.857e-06
  Validacioni NLL: 16.54813317
KANDIDAT: R=2, lambda=3.0, validacioni NLL=16.54839546

IZABRANO: R=2, lambda=3.0, validacioni NLL=16.54839546
loto7_4710_k86.csv / odvojeni test: R=2, lambda=3.0, gubitak=16.54750373, max|grad|=1.267e-06
ODVOJENI TEST: NLL=16.54729139, kola=942
Pogodaka 0–7: 207 376 272 74 13 0 0 0
Test se ne koristi za promenu parametara.
loto7_4710_k86.csv / svih 4709 prelaza: R=2, lambda=3.0, gubitak=16.54741600, max|grad|=2.900e-07
loto7_4710_k86_loto_2977.csv / validacija 1: R=1, lambda=0.03, gubitak=16.53080310, max|grad|=8.959e-07
  Validacioni NLL: 16.55326573
loto7_4710_k86_loto_2977.csv / validacija 2: R=1, lambda=0.03, gubitak=16.53241864, max|grad|=6.394e-07
  Validacioni NLL: 16.56083223
KANDIDAT: R=1, lambda=0.03, validacioni NLL=16.55704262
loto7_4710_k86_loto_2977.csv / validacija 1: R=1, lambda=0.3, gubitak=16.53977096, max|grad|=2.616e-07
  Validacioni NLL: 16.54868150
loto7_4710_k86_loto_2977.csv / validacija 2: R=1, lambda=0.3, gubitak=16.54038890, max|grad|=2.157e-07
  Validacioni NLL: 16.55562933
KANDIDAT: R=1, lambda=0.3, validacioni NLL=16.55214958
loto7_4710_k86_loto_2977.csv / validacija 1: R=1, lambda=3.0, gubitak=16.54630306, max|grad|=3.362e-08
  Validacioni NLL: 16.54746418
loto7_4710_k86_loto_2977.csv / validacija 2: R=1, lambda=3.0, gubitak=16.54645471, max|grad|=3.468e-08
  Validacioni NLL: 16.54957508
KANDIDAT: R=1, lambda=3.0, validacioni NLL=16.54851786
loto7_4710_k86_loto_2977.csv / validacija 1: R=2, lambda=0.03, gubitak=16.52860004, max|grad|=4.679e-07
  Validacioni NLL: 16.55633959
loto7_4710_k86_loto_2977.csv / validacija 2: R=2, lambda=0.03, gubitak=16.53077247, max|grad|=4.262e-07
  Validacioni NLL: 16.56174354
KANDIDAT: R=2, lambda=0.03, validacioni NLL=16.55903702
loto7_4710_k86_loto_2977.csv / validacija 1: R=2, lambda=0.3, gubitak=16.53945046, max|grad|=8.770e-07
  Validacioni NLL: 16.54904692
loto7_4710_k86_loto_2977.csv / validacija 2: R=2, lambda=0.3, gubitak=16.54015153, max|grad|=6.903e-07
  Validacioni NLL: 16.55575746
KANDIDAT: R=2, lambda=0.3, validacioni NLL=16.55239655
loto7_4710_k86_loto_2977.csv / validacija 1: R=2, lambda=3.0, gubitak=16.54625677, max|grad|=3.263e-07
  Validacioni NLL: 16.54749708
loto7_4710_k86_loto_2977.csv / validacija 2: R=2, lambda=3.0, gubitak=16.54641975, max|grad|=1.738e-07
  Validacioni NLL: 16.54959293
KANDIDAT: R=2, lambda=3.0, validacioni NLL=16.54854325

IZABRANO: R=1, lambda=3.0, validacioni NLL=16.54851786
loto7_4710_k86_loto_2977.csv / odvojeni test: R=1, lambda=3.0, gubitak=16.54681965, max|grad|=1.244e-08
ODVOJENI TEST: NLL=16.54777444, kola=596
Pogodaka 0–7: 129 242 163 52 10 0 0 0
Test se ne koristi za promenu parametara.
loto7_4710_k86_loto_2977.csv / svih 2976 prelaza: R=1, lambda=3.0, gubitak=16.54692208, max|grad|=3.421e-09
loto7_4710_k86_loto_plus_1733.csv / validacija 1: R=1, lambda=0.03, gubitak=16.51185937, max|grad|=2.737e-07
  Validacioni NLL: 16.56345846
loto7_4710_k86_loto_plus_1733.csv / validacija 2: R=1, lambda=0.03, gubitak=16.51725833, max|grad|=8.573e-07
  Validacioni NLL: 16.56252199
KANDIDAT: R=1, lambda=0.03, validacioni NLL=16.56299022
loto7_4710_k86_loto_plus_1733.csv / validacija 1: R=1, lambda=0.3, gubitak=16.53067093, max|grad|=7.377e-08
  Validacioni NLL: 16.55498150
loto7_4710_k86_loto_plus_1733.csv / validacija 2: R=1, lambda=0.3, gubitak=16.53346951, max|grad|=3.895e-07
  Validacioni NLL: 16.55911771
KANDIDAT: R=1, lambda=0.3, validacioni NLL=16.55704960
loto7_4710_k86_loto_plus_1733.csv / validacija 1: R=1, lambda=3.0, gubitak=16.54392966, max|grad|=4.544e-08
  Validacioni NLL: 16.54815370
loto7_4710_k86_loto_plus_1733.csv / validacija 2: R=1, lambda=3.0, gubitak=16.54467114, max|grad|=5.167e-08
  Validacioni NLL: 16.54987791
KANDIDAT: R=1, lambda=3.0, validacioni NLL=16.54901580
loto7_4710_k86_loto_plus_1733.csv / validacija 1: R=2, lambda=0.03, gubitak=16.50862483, max|grad|=2.794e-07
  Validacioni NLL: 16.56008932
loto7_4710_k86_loto_plus_1733.csv / validacija 2: R=2, lambda=0.03, gubitak=16.51398181, max|grad|=2.594e-07
  Validacioni NLL: 16.56576067
KANDIDAT: R=2, lambda=0.03, validacioni NLL=16.56292500
loto7_4710_k86_loto_plus_1733.csv / validacija 1: R=2, lambda=0.3, gubitak=16.53022057, max|grad|=3.020e-07
  Validacioni NLL: 16.55435615
loto7_4710_k86_loto_plus_1733.csv / validacija 2: R=2, lambda=0.3, gubitak=16.53300222, max|grad|=6.482e-07
  Validacioni NLL: 16.55954892
KANDIDAT: R=2, lambda=0.3, validacioni NLL=16.55695254
loto7_4710_k86_loto_plus_1733.csv / validacija 1: R=2, lambda=3.0, gubitak=16.54386322, max|grad|=6.365e-07
  Validacioni NLL: 16.54805392
loto7_4710_k86_loto_plus_1733.csv / validacija 2: R=2, lambda=3.0, gubitak=16.54460211, max|grad|=4.136e-07
  Validacioni NLL: 16.54993613
KANDIDAT: R=2, lambda=3.0, validacioni NLL=16.54899502

IZABRANO: R=2, lambda=3.0, validacioni NLL=16.54899502
loto7_4710_k86_loto_plus_1733.csv / odvojeni test: R=2, lambda=3.0, gubitak=16.54515821, max|grad|=4.027e-07
ODVOJENI TEST: NLL=16.54848202, kola=347
Pogodaka 0–7: 80 141 95 25 4 2 0 0
Test se ne koristi za promenu parametara.
loto7_4710_k86_loto_plus_1733.csv / svih 1732 prelaza: R=2, lambda=3.0, gubitak=16.54579713, max|grad|=6.238e-07

ZAVRŠNI REZULTATI TERMODINAMIKA V1
Energija, slobodna energija i entropija pripadaju matematičkom modelu.
Modelna verovatnoća nije potvrđena verovatnoća stvarnog izvlačenja.

CSV: loto7_4710_k86.csv
NEXT: 08 x 22 y 26 z 37
Izabrani domet interakcije: 2
Izabrana regularizacija: 3.0
Validacioni NLL: 16.54839546
Odvojeni test NLL: 16.54729139
Prosečno pogodaka na odvojenom testu: 1.267516
Test pogodaka 0–7: 207 376 272 74 13 0 0 0
Modelna verovatnoća NEXT: 7.08148946537e-08
log Z: 16.5489907115
Najmanja energija NEXT: -0.0857942296
Srednja energija U: -0.0005903001
Slobodna energija F: -16.5489907115
Entropija S: 16.5484004114
Zbir uključnih verovatnoća: 7.000000000000
Promena uključnih verovatnoća kroz kontekst: 1.534182e-04

CSV: loto7_4710_k86_loto_2977.csv
NEXT: 08 x 21 y 23 z 35
Izabrani domet interakcije: 1
Izabrana regularizacija: 3.0
Validacioni NLL: 16.54851786
Odvojeni test NLL: 16.54777444
Prosečno pogodaka na odvojenom testu: 1.281879
Test pogodaka 0–7: 129 242 163 52 10 0 0 0
Modelna verovatnoća NEXT: 7.19158969537e-08
log Z: 16.5491598687
Najmanja energija NEXT: -0.1013913701
Srednja energija U: -0.0008585716
Slobodna energija F: -16.5491598687
Entropija S: 16.5483012971
Zbir uključnih verovatnoća: 7.000000000000
Promena uključnih verovatnoća kroz kontekst: 2.064195e-04

CSV: loto7_4710_k86_loto_plus_1733.csv
NEXT: 02 x 08 y 23 z 34
Izabrani domet interakcije: 2
Izabrana regularizacija: 3.0
Validacioni NLL: 16.54899502
Odvojeni test NLL: 16.54848202
Prosečno pogodaka na odvojenom testu: 1.244957
Test pogodaka 0–7: 80 141 95 25 4 2 0 0
Modelna verovatnoća NEXT: 7.44758653326e-08
log Z: 16.5486180579
Najmanja energija NEXT: -0.1358273386
Srednja energija U: -0.0005184966
Slobodna energija F: -16.5486180579
Entropija S: 16.5480995613
Zbir uključnih verovatnoća: 7.000000000000
Promena uključnih verovatnoća kroz kontekst: 3.904580e-04

NEXT je tačno stanje najmanje naučene energije.
Direktne interakcije postoje samo unutar izabranog dometa 1 ili 2.
Prediktivna prednost zahteva procenu prikazanih rezultata odvojenog testa.
"""



"""
Verovatnoća i termodinamika povezane su preko statističke mehanike i entropije. 
Zakoni verovatnoće važe šire; termodinamika ih koristi da poveže mikrostanja sa osobinama sistema.

Za Loto 7/39 istražiti raspodelu:
\[
P(S\mid H)=
\frac{\exp[-\beta E_\theta(S,H)]}
{\sum_{|T|=7}\exp[-\beta E_\theta(T,H)]}.
\]
Ovde je \(S\) kombinacija, \(H\) prethodna istorija, a \(E_\theta\) naučena funkcija energije. 
Parametar \(\beta\) određuje koliko je raspodela koncentrisana.

Definisati i naučiti energiju iz istorijskih prelaza. 
Sam naziv „energija”, temperatura ili entropija ne daju predikciju. 
Za loto bi to bio matematički model inspirisan termodinamikom, čiju vrednost proveravam hronološki.

V1 će koristiti Gibbsovu raspodelu sa naučenom energijom pojedinačnih brojeva i njihovih parova. 
Za tačnu normalizaciju koristiću transfer-matricu, a NEXT će biti stanje najmanje energije. 
Interakcije će biti ograničene na udaljenost 1 ili 2 u numeraciji; nijedna legalna kombinacija time se ne isključuje.

Ugrađene provere obuhvataju i izvode, entropiju, granične slučajeve i maksimum;
"""
