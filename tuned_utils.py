from __future__ import annotations
import numpy as np
import pandas as pd

from kmeans import KMeansAlgo
from guha_streaming_new import Guha_Stream_KMeans
from ailon_streaming_new import Ailon_Coreset
from boutsidis_streaming_new import Boutsidis_Streaming
from charikar_streaming_new import Charikar_KMeans
from guha_tuning import tune_guha_parameters
from ailon_tuning import tune_ailon_parameters
from boutsidis_tuning import tune_boutsidis_parameters
from charikar_tuned import tune_charikar_parameters
from results import Algo
from sklearn.datasets import make_blobs

def tuned_algorithms(
    seeds=(42, 77, 211),
    quality_weight=0.5,
    runtime_weight=0.25,
    memory_weight=0.25
) -> list[Algo]:
    """
    Return the best parameters for each algorithm, based on the tuning results.

    Priority:
    1) NMI
    2) ARI
    3) negative SSE
    """ 
    d = 10
    n = 10_000
    X, y = make_blobs(n_samples=n, centers=8, n_features=2, cluster_std=1.5, random_state=321)
    A = np.array([[0.6, -0.8], [0.4, 0.9]])
    X = X @ A
    rng = np.random.default_rng(123)
    X = np.hstack([X, rng.normal(0, 0.1, size=(n, d - 2))])

    algos = [KMeansAlgo(max_iter=300)]
    ailon_df = tune_ailon_parameters(
        samples=X,
        k=8,
        output_dir="output/ailon_tuned",
        labels=y,
        coreset_factors=(1.0, 1.5, 2.0),
        repeat_factors=(0.75, 1.0, 1.5),
        seeds=seeds,
        quality_weight=quality_weight,
        runtime_weight=runtime_weight,
        memory_weight=memory_weight,
    )
    algos.append(Ailon_Coreset(chunk_size=8192,coreset_factor=ailon_df.iloc[0]["coreset_factor"],
                                repeat_factor=ailon_df.iloc[0]["repeat_factor"]))
    print(f"The best parameters for Ailon are: coreset_factor={ailon_df.iloc[0]['coreset_factor']}, repeat_factor={ailon_df.iloc[0]['repeat_factor']}")
    boutsidis_df = tune_boutsidis_parameters(
        samples=X,
        k=8,
        r_min=2,
        output_dir="output/boutsidis_tuned",
        labels=y,
        eps_values = (1.5, 2.5, 3.5),
        c2_values = (1.0, 2.0, 3.0),
        seeds=seeds,
        quality_weight=quality_weight,
        runtime_weight=runtime_weight,
        memory_weight=memory_weight,
    )
    print(f"The best parameters for Boutsidis are: eps={boutsidis_df.iloc[0]['eps']}, c2={boutsidis_df.iloc[0]['c2']}")
    algos.append(Boutsidis_Streaming(eps=boutsidis_df.iloc[0]["eps"], c2=boutsidis_df.iloc[0]["c2"], chunk_size=8192))

    guha_df = tune_guha_parameters(
        samples=X,
        k=14,
        output_dir="output/guha_tuned",
        labels=y,
        chunk_size=4096,
        m_factor_values=(1.0, 2.0, 3.0, 4.0, 5.0),
        seeds=seeds,
        quality_weight=quality_weight,
        runtime_weight=runtime_weight,
        memory_weight=memory_weight,
    )
    print(f"The best parameter for Guha is: m_factor={guha_df.iloc[0]['m_factor']}")
    algos.append(Guha_Stream_KMeans(chunk_size=8192, m_factor=guha_df.iloc[0]["m_factor"]))

    charikar_df = tune_charikar_parameters(
        samples=X,
        k=8,
        output_dir="output/charikar_tuned",
        labels=y,
        chunk_size=4092,
        beta_values = (1.5, 3, 5),
        gamma_values = (0.25, 0.5, 1),
        seeds=seeds,
        quality_weight=quality_weight,
        runtime_weight=runtime_weight,
        memory_weight=memory_weight,
    )
    print(f"The best parameters for Charikar are: beta={charikar_df.iloc[0]['beta']}, gamma={charikar_df.iloc[0]['gamma']}")
    algos.append(Charikar_KMeans(beta=charikar_df.iloc[0]["beta"], gamma=charikar_df.iloc[0]["gamma"], chunk_size=8192))
    return algos

if __name__ == "__main__":
    algo = tuned_algorithms()
    print("done tuning the algorithms")