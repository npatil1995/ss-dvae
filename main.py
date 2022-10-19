from pathlib import Path
import shutil
import warnings


import pandas as pd
import numpy as np
import configargparse
import os

from argparse import ArgumentParser

from dvae import run_dvae



if __name__ == "__main__":
    # Filter two pyro warnings
    warnings.filterwarnings("ignore", message=".*was not registered in the param store because requires_grad=False.*")
    warnings.filterwarnings("ignore", message=".*torch.tensor results are registered as constants in the trace.*")

    parser = configargparse.ArgParser(
        description="parse args",
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )


    parser.add("-c", "--config", is_config_file=True, default=None)
    parser.add("--input_dir", default=None)


    # parser.add("--output_dir", default="/output")
    parser.add("--output_dir", default="E:/thesis papers temp/dvae-main/dvae-main/output")
    parser.add("--temp_output_dir", default=None, help="Temporary model storage during run, when I/O bound")

    parser.add("--train_path", default="train_data_updated.npz")
    parser.add("--eval_path", default="valid_data_updated.npz")
    parser.add("--vocab_path", default="vocab_updated.json")
    parser.add("--topic_word_init_path", default=None)
    parser.add("--topic_word_prior_path", default=None)
    parser.add("--to_dense", default=False, action="store_true")

    #was None changed to 5
    parser.add("--num_topics", default=50, type=int)

    parser.add("--encoder_embeddings_dim", default=100, type=int)
    parser.add("--encoder_hidden_dim", default=0, type=int)
    parser.add("--dropout", default=0.25, type=float)
    parser.add("--alpha_prior", default=0.01, type=float)
    parser.add("--decoder_bias", default=True, action="store_true")
    parser.add("--no_decoder_bias", dest="decoder_bias", action="store_false")
    parser.add("--softmax_beta", default=False, action="store_true")

    parser.add('--learning_rate', default=0.001, type=float)
    parser.add('--topic_word_regularization', default=None, type=float)
    parser.add('--adam_beta_1', default=0.9, type=float)
    parser.add('--adam_beta_2', default=0.999, type=float)

    parser.add("--batch_size", default=200, type=int)

    # was 200 changed to 5 to 1
    parser.add("--num_epochs", default= 1, type=int)
    parser.add("--epochs_to_anneal_bn", default=0, type=int)
    parser.add("--epochs_to_anneal_kl", default=100, type=int)
    
    parser.add("--eval_words", default=10, type=int)

    #was 50 changed to 10
    parser.add("--topic_words_to_save", default=10, type=int)

    #was npmi changed to none
    parser.add("--target_metric", default= None, choices=["npmi", "loss", "tu"])
    parser.add("--compute_eval_loss", default=False, action="store_true")
    parser.add("--max_acceptable_overlap", type=int, default=None)
    parser.add("--eval_step", default=1, type=int)

    parser.add("--run_seeds", default=[42], type=int, nargs="+", help="Seeds to use for each run")
    parser.add('--gpu', action='store_true', default=False, help='whether to use cuda')
    parser.add('--jit', action='store_true', default=False, help='whether to use PyTorch jit')
    args = parser.parse_args()

    # Filter two pyro warnings

    # Run for each seedfrom argparse import ArgumentParser
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(exist_ok=True, parents=True)

    for i, seed in enumerate(args.run_seeds):
        # make subdirectories for each run
        if len(args.run_seeds) == 1:
            output_dir = base_output_dir
        else:
            output_dir = Path(base_output_dir, str(seed))
            output_dir.mkdir(exist_ok=True, parents=True)

        run_dir = str(output_dir)
        if args.temp_output_dir:
            run_dir = Path(args.temp_output_dir, str(np.random.randint(1000)))
    
        # train
        print(f"\nOn run {i+1} of {len(args.run_seeds)}")
        model, metrics = run_dvae(
            input_dir=args.input_dir,
            output_dir=run_dir,
            train_path=args.train_path,
            eval_path=args.eval_path,
            vocab_path=args.vocab_path,
            topic_word_init_path=args.topic_word_init_path,
            topic_word_prior_path=args.topic_word_prior_path,
            num_topics=args.num_topics,
            to_dense=args.to_dense,
            encoder_embeddings_dim=args.encoder_embeddings_dim,
            encoder_hidden_dim=args.encoder_hidden_dim,
            dropout=args.dropout,
            alpha_prior=args.alpha_prior,
            decoder_bias=args.decoder_bias,
            softmax_beta=args.softmax_beta,
            learning_rate=args.learning_rate,
            topic_word_regularization=args.topic_word_regularization,
            adam_beta_1=args.adam_beta_1,
            adam_beta_2=args.adam_beta_2,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            epochs_to_anneal_bn=args.epochs_to_anneal_bn,
            epochs_to_anneal_kl=args.epochs_to_anneal_kl,
            eval_words=args.eval_words,
            topic_words_to_save=args.topic_words_to_save,
            target_metric=args.target_metric,
            compute_eval_loss=args.compute_eval_loss,
            max_acceptable_overlap=args.max_acceptable_overlap,
            eval_step=args.eval_step,
            seed=seed,
            gpu=args.gpu,
        )
        print("model", model)
        print("metrics",metrics)
        if args.temp_output_dir:
            shutil.copytree(run_dir, output_dir, dirs_exist_ok=True)
            shutil.rmtree(run_dir)
    
    # Aggregate results
    agg_run_results = []
    for seed in args.run_seeds:
        if len(args.run_seeds) == 1:
            output_dir = base_output_dir
        else:
            output_dir = Path(base_output_dir, str(seed))
        results = pd.read_csv(Path(output_dir, "results.csv"))
        agg_run_results.append({
            "seed": seed,
            "best_npmi": np.max(results.npmi),
            "best_npmi_epoch": np.argmax(results.npmi),
            "best_tu_at_best_npmi": results.tu[np.argmax(results.npmi)],
            "best_to_at_best_npmi": results.to[np.argmax(results.npmi)],
            "overlaps_at_best_npmi": results.complete_overlaps[np.argmax(results.npmi)],
        })

    agg_run_results_df = pd.DataFrame.from_records(agg_run_results)
    agg_run_results_df.to_csv(Path(base_output_dir, "run_results.csv"))
    print(
        f"\n=== Results over {len(args.run_seeds)} runs ===\n"
        f"Mean NPMI: "
        f"{agg_run_results_df.best_npmi.mean():0.4f} ({agg_run_results_df.best_npmi.std():0.4f}) "
        f"@ epoch {np.mean(agg_run_results_df.best_npmi_epoch):0.1f} / {args.num_epochs}\n"
        f"Mean best TU @ best NPMI: {agg_run_results_df.best_tu_at_best_npmi.mean():0.4f}\n"
        f"Mean best TO @ best NPMI: {agg_run_results_df.best_to_at_best_npmi.mean():0.4f}"
    )