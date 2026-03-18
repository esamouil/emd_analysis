import plotly.io as pio

pub_clean_template = dict(
    layout=dict(
        # Figure
        width=600,   # ~5 inch at 120 dpi
        height=600,
        paper_bgcolor="white",
        plot_bgcolor="white",

        # Fonts
        font=dict(
            family="sans-serif",
            size=12,
            color="black"
        ),

        # Title
        title=dict(
            font=dict(size=12)
        ),

        # Axes
        xaxis=dict(
            showgrid=False,
            showline=True,
            linewidth=1.2,
            linecolor="black",
            ticks="outside",
            tickwidth=1.2,
            ticklen=5,
            tickfont=dict(size=10),
            title=dict(font=dict(size=12))
        ),
        yaxis=dict(
            showgrid=False,
            showline=True,
            linewidth=1.2,
            linecolor="black",
            ticks="outside",
            tickwidth=1.2,
            ticklen=5,
            tickfont=dict(size=10),
            title=dict(font=dict(size=12))
        ),

        # Legend
        legend=dict(
            font=dict(size=10),
            borderwidth=0
        ),

        # Color cycle
        colorway=[
            "black",
            "#1f77b4",  # tab:blue
            "#d62728",  # tab:red
            "#2ca02c",  # tab:green
            "#ff7f0e",  # tab:orange
            "#9467bd",  # tab:purple
        ],
    ),

    data=dict(
        scatter=[
            dict(
                line=dict(width=1.6),
                marker=dict(size=4)
            )
        ]
    )
)

pio.templates["pub_clean"] = pub_clean_template
