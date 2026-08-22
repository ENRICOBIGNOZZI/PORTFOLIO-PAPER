def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    FIGURES.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)

    raw = load_all()
    summ = summary(raw)
    rates = rate_table(raw, args.bootstrap, args.seed)
    history = history_required(summ)

    raw.to_csv(RESULTS / "mc_symmetric_all_raw.csv", index=False)
    summ.to_csv(RESULTS / "mc_symmetric_curve_summary.csv", index=False)
    rates.to_csv(RESULTS / "mc_symmetric_rate_summary.csv", index=False)
    history.to_csv(RESULTS / "mc_symmetric_history_required.csv", index=False)

    save_model_matrix(summ, "Linear", rates)
    save_model_matrix(summ, "Sobolev", rates)
    for design in ("SNR", "N", "d"):
        save_history_figure(history, design)
    write_main_table(rates, history)

    metadata = {
        "models": ["Linear", "Sobolev"],
        "comparative_statics": ["SNR", "N", "d"],
        "history_thresholds": list(THRESHOLDS),
        "rate_estimator": "six largest T cells; median relative squared-Sharpe shortfall; cell bootstrap",
        "linear_theory_rate": 1.0,
        "sobolev_theory_rate_snr_and_N": 2.0 / 3.0,
        "sobolev_dimension_rates": {str(d): 6.0 / (6.0 + d) for d in D_LEVELS},
        "notes": [
            "SNR and N change the onset of the fixed-model asymptotic law, not its structural exponent.",
            "State dimension changes the Sobolev exponent but not the fixed-dimensional parametric exponent.",
            "Unreached history thresholds are reported as lower bounds at the largest simulated T.",
        ],
    }
    (RESULTS / "mc_symmetric_postprocess_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print("\nRATE SUMMARY")
    print(rates.to_string(index=False))
    print("\nHISTORY REQUIRED")
    print(history[history["threshold"].isin([0.75, 0.90])].to_string(index=False))


if __name__ == "__main__":
    main()
