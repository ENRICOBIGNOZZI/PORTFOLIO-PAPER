def run_tasks_incremental(task, T_grid: tuple[int, ...], reps_by_T: dict[int, int],
                          seed: int, jobs: int, output: Path,
                          sort_cols: list[str], resume: bool) -> pd.DataFrame:
    existing = pd.read_csv(output) if (resume and output.exists()) else pd.DataFrame()
    completed = set(existing["T"].unique().tolist()) if not existing.empty else set()
    blocks = [existing] if not existing.empty else []
    for T in T_grid:
        if T in completed:
            print(f"skip completed T={T} -> {output.name}", flush=True)
            continue
        reps = reps_by_T[T]
        print(f"run {output.name}: T={T}, reps={reps}", flush=True)
        task_blocks = Parallel(n_jobs=jobs, backend="threading", verbose=0)(
            delayed(task)(T, rep, seed) for rep in range(reps)
        )
        rows: list[dict] = []
        for block in task_blocks:
            rows.extend(block)
        blocks.append(pd.DataFrame(rows))
        current = pd.concat(blocks, ignore_index=True).sort_values(sort_cols)
        current.to_csv(output, index=False)
    return pd.concat(blocks, ignore_index=True).sort_values(sort_cols)

