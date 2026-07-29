
RANDOM_SEED = 42


def benchmark_tsne():

    import os
    import time
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from memory_profiler import memory_usage
    from sklearn.manifold import TSNE
    from openTSNE import TSNE as FastTSNE

    save_path = './results/tsne_benchmark'
    os.makedirs(save_path, exist_ok=True)

    rng = np.random.RandomState(RANDOM_SEED)

    run = True
    if run:

        def measure_block(func):

            start = time.time()
            mem_usage, out = memory_usage(
                func,
                interval=0.01,
                timestamps=False,
                retval=True,
            )
            end = time.time()

            peak_mem_gb = max(mem_usage) / 1024  # MB to GB
            wall_time = end - start

            return peak_mem_gb, wall_time, out

        def run_tsne_scikit_learn(x: np.ndarray):
            reducer = TSNE(n_jobs=8, random_state=RANDOM_SEED)
            x_tsne = reducer.fit_transform(X=x)
            return x_tsne

        def run_tsne_opentsne(x: np.ndarray):
            reducer = FastTSNE(n_jobs=8, random_state=RANDOM_SEED)
            x_tsne = reducer.fit(X=x)
            return x_tsne


        n_features = [10, ]
        sample_sizes = [100, 1000, 10000, 20000, 30000, 40000, 50000]
        methods = ['Scikit-learn', 'openTSNE']
        method_to_fct = {
            'Scikit-learn': run_tsne_scikit_learn,
            'openTSNE': run_tsne_opentsne,
        }

        # Warmup run
        reducer = FastTSNE(n_jobs=8, random_state=RANDOM_SEED)
        reducer.fit(X=rng.randn(100, 10))

        rows = []
        for n in n_features:

            print(f'# --- Num features: {n} --- ')

            for sample_size in sample_sizes:

                print(f'# --- Sample size: {sample_size} ')

                x_test = rng.randn(sample_size, n)

                for method in methods:

                    print(f'# --- {method}')

                    def tsne_block():
                        return method_to_fct[method](x_test)

                    mem_peak, wall_time, out = measure_block(tsne_block)

                    rows.append({
                        'method': method,
                        'n_features': n,
                        'n_samples': sample_size,
                        'mem_peak_gb': mem_peak,
                        'wall_time_s': wall_time,
                    })

            res_df = pd.DataFrame(rows)

            res_df.to_csv(os.path.join(save_path, 'res_df.csv'), index=False)

            print(res_df)

        res_df = pd.read_csv(os.path.join(save_path, 'res_df.csv'))

        fig, ax = plt.subplots(dpi=600)
        sns.lineplot(
            data=res_df,
            x='n_samples',
            y='mem_peak_gb',
            hue='method',
            palette='Set2',
            marker='o',
            ax=ax,
        )
        ax.set_xlabel('Sample size')
        ax.set_ylabel('Peak RSS [GB]')
        ax.set_xscale('log')
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(save_path, 'peak_memory.png'))
        plt.close(fig)

        fig, ax = plt.subplots(dpi=600)
        sns.lineplot(
            data=res_df,
            x='n_samples',
            y='wall_time_s',
            hue='method',
            palette='Set2',
            marker='o',
            ax=ax,
        )
        ax.set_xlabel('Sample size')
        ax.set_ylabel('Wall time [s]')
        ax.set_xscale('log')
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(save_path, 'wall_time.png'))
        plt.close(fig)


def minimal_example():

    import numpy as np
    from openTSNE import TSNE

    x_train = np.random.randn(100, 10)

    reducer = TSNE(n_jobs=-1)
    x_tsne = reducer.fit(X=x_train)

    print(x_tsne)


if __name__ == '__main__':

    # INSTALLATION:
    # conda install opentsne

    # benchmark_tsne()

    # minimal_example()

    print('done')















