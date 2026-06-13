import pandas as pd
import matplotlib.pyplot as plt

def plot(agent_name):
    df = pd.read_csv(f"results/{agent_name}/{agent_name}.csv")

    fig, ax = plt.subplots(figsize=(10, 5))

    # Raw reward — faint
    ax.plot(df['episode'], df['reward'], alpha=0.3, linewidth=0.8, color='steelblue', label='Raw Reward')

    # Rolling average — bold on top
    ax.plot(df['episode'], df['reward'].rolling(100).mean(), linewidth=2, color='steelblue', label='Rolling Average (100)')

    ax.set_xlabel('Episode', fontsize=13)
    ax.set_ylabel('Total Reward per Episode', fontsize=13)
    ax.set_title(f'{agent_name} — Reward over Episodes', fontsize=15)
    ax.legend(fontsize=11)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"results/{agent_name}/{agent_name}.pdf", dpi=300)
    plt.close()