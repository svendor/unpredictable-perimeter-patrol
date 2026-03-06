from LSR.lsr_cycle import LSRCycle
from strategy import Strategy

import jax
import jax.numpy as jnp
import jax.experimental.sparse as sparse
from jax import lax, config
import optax
config.update("jax_debug_nans", True)

import numpy as np

# --- 1. DATA PREPARATION (Run once) ---
def prepare_jax_data(template: list[list[tuple[int, int]]]):
    rows_list, cols_list, param_idx_list = [], [], []

    for p_idx, equivalents in enumerate(template):
        for r, c in equivalents:
            rows_list.append(r)
            cols_list.append(c)
            param_idx_list.append(p_idx)

    return (jnp.array(rows_list, dtype=jnp.int32),
            jnp.array(cols_list, dtype=jnp.int32),
            jnp.array(param_idx_list, dtype=jnp.int32))

# --- 2. THE DIFFERENTIABLE MODEL ---
@jax.jit(static_argnames=('n_states'))
def build_matrix(params, rows, cols, param_idx, n_states):
    # Shift params by the max value used in each row to prevent overflow
    # This keeps the exp() values <= 1.0
    row_max = jax.ops.segment_max(params[param_idx], rows, num_segments=n_states)
    shifted_logits = params[param_idx] - row_max[rows]

    raw_vals = jnp.exp(shifted_logits)
    row_sums = jax.ops.segment_sum(raw_vals, rows, num_segments=n_states)

    norm_vals = raw_vals / (row_sums[rows] + 1e-12)
    return sparse.BCOO((norm_vals, jnp.stack([rows, cols], axis=1)), shape=(n_states, n_states))

@jax.jit(static_argnames=('n_states', 'num_histories'))
def objective_fn(params, rows, cols, param_idx, n_states, num_histories):
    # 1. Build the matrix
    T = build_matrix(params, rows, cols, param_idx, n_states)

    # 2. Create the batch of starting distributions
    # Shape: (num_histories, n_states). Row i has 1.0 at index i.
    start_dists = jnp.eye(n_states)[:num_histories]

    # Track the minimum return probability for each history independently
    init_mins = jnp.ones(num_histories)

    mask = jnp.ones(n_states).at[:num_histories].set(0.0)

    def step(carry, _):
        # dists is now (num_histories, n_states)
        dists, current_mins = carry

        # 3. Dense-Sparse Multiplication (The Matrix-Matrix version)
        # Using T (row-stochastic) with dists (as rows)
        # Result: (num_histories, n_states)
        next_dists = dists @ T

        # 4. Calculate return probabilities for the whole batch
        # Sum mass in columns 0...num_histories-1 for each row
        ret_probs = jnp.sum(next_dists[:, :num_histories], axis=1)
        new_mins = jnp.minimum(current_mins, ret_probs)

        # 5. Batch Renormalization
        masked_dists = next_dists * mask
        total_masses = jnp.sum(masked_dists, axis=1, keepdims=True)

        # Guard against 0/0
        safe_denoms = jnp.where(total_masses > 0, total_masses, 1.0)
        next_dists = jnp.where(total_masses > 0, masked_dists / safe_denoms, 0.0)

        return (next_dists, new_mins), None

    # Run the loop for all 9 starting states simultaneously
    (_, all_lowest_mins), _ = lax.scan(step, (start_dists, init_mins), jnp.arange(15))

    # Find the worst-case history (the one with the minimum return probability)
    worst_case_min = jnp.min(all_lowest_mins)

    # Return negative to maximize
    return -worst_case_min


def optimize(template: list[list[tuple[int, int]]], n_states: int, num_histories: int, num_iterations: int = 200):
    rows, cols, p_idx = prepare_jax_data(template)

    # A. Define Optimizer
    lr_schedule = optax.exponential_decay(
        init_value=0.05,
        transition_steps=50,
        decay_rate=0.5
    )

    # 2. Chain the schedule with Gradient Clipping
    # Clipping prevents "exploding" updates if a gradient spike occurs
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr_schedule)
    )

    # B. Initialize Parameters and Optimizer State
    params = jnp.zeros(n_states * 2)
    opt_state = optimizer.init(params)

    # C. Define the Update Step
    @jax.jit(static_argnames=('n_states', 'num_histories'))
    def update_step(params, opt_state, rows, cols, p_idx, n_states, num_histories):
        loss, grads = jax.value_and_grad(objective_fn)(params, rows, cols, p_idx, n_states, num_histories)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # D. Run the Loop
    for step in range(num_iterations + 1):
        params, opt_state, loss = update_step(params, opt_state, rows, cols, p_idx, n_states, num_histories)

        if step % 20 == 0:
            print(f"Iteration {step}: Min Probability = {loss:.6f}")

    # E. Extract Final Result
    print("\nOptimization complete.")
    return build_matrix(params, rows, cols, p_idx, n_states)

def print_human_readable(transition_matrix, strategy: Strategy):
    """Take the optimized parameters and print them in a human-readable format."""
    rows, cols, params = transition_matrix.indices[:, 0], transition_matrix.indices[:, 1], transition_matrix.data

    print("\nOptimized Transition Probabilities from location 0:")

    # We know that, by symmetry, we only need to consider location 0 and all their histories.
    # In other words, only the rows indexed by 0, 1, ..., num_histories-1 are relevant.
    moves = np.zeros((num_histories, 3), dtype=float)

    for i, j, parameter in zip(rows, cols, params):
        if i < num_histories:
            next_loc = j // num_histories
            # next_loc is n-1, 0 or 1, if the moves were Left Same or Right. By adding 1 and performing modulo
            # we obtain 0, 1, 2.
            move_index = (next_loc + 1) % n
            moves[i, move_index] = parameter

    for i in range(num_histories):
        print(f"{strategy.get_history_string(i)}: {moves[i]}")

# --- 3. OPTAX OPTIMIZATION LOOP ---
if __name__ == "__main__":
    from cli import parse_args

    args = parse_args()
    n = args.n
    m = args.m

    strategy = LSRCycle(n, m)
    my_template = strategy.get_transition_template()
    num_histories = strategy.num_histories
    num_states = n*num_histories
    print(f"Number of histories: {num_histories}")
    print(f"Number of states: {num_states}")

    transition_matrix = optimize(my_template, num_states, num_histories, num_iterations=400)
    print_human_readable(transition_matrix, strategy)