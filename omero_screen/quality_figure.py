import matplotlib.pyplot as plt


def quality_control_fig(df):
    """Plot the quality control data for each image"""
    df["position"] = df["position"].astype("category")
    medians = (
        df.groupby(["position", "channel"], observed=False)["intensity_median"].mean().reset_index()
    )
    std = df.groupby(["position", "channel"], observed=False)["intensity_median"].std().reset_index()
    channel_num = len(df.channel.unique())
    well_num = len(df.position.unique())
    # Plotting the results
    fig, ax = plt.subplots(nrows=channel_num, figsize=(well_num, channel_num))
    for i, channel in enumerate(df.channel.unique()):
        channel_df = medians[medians["channel"] == channel]
        channel_std = std[std["channel"] == channel]["intensity_median"]
        y_min = (channel_df["intensity_median"] - channel_std).min()
        y_max = (channel_df["intensity_median"] + channel_std).max()
        padding = (y_max - y_min) * 0.1  # 10% padding
        ax[i].errorbar(
            channel_df["position"],
            channel_df["intensity_median"],
            yerr=channel_std,
            fmt="o",
        )
        ax[i].set_title(channel)
        ax[i].set_xticks(range(len(channel_df["position"])))
        ax[i].set_xticklabels(channel_df["position"])
        ax[i].set_xlim(-0.5, len(channel_df["position"]) - 0.5)
        ax[i].set_ylim(y_min - padding, y_max + padding)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("/Users/hh65/Desktop/test_plate01/test_plate01_quality_ctr.csv")
    quality_control_fig(df)
