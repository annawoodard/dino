import matplotlib.pyplot as plt
import seaborn as sns
CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'

color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
              CB91_Purple, CB91_Violet]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

sns.set(
    # font="Franklin Gothic Book",
    rc={
        'axes.prop_cycle': plt.cycler(color=color_list),
        "axes.axisbelow": False,
        "axes.edgecolor": "lightgrey",
        "axes.facecolor": "None",
        "axes.grid": False,
        "axes.labelcolor": "dimgrey",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "figure.facecolor": "white",
        "lines.solid_capstyle": "round",
        "patch.edgecolor": "w",
        "patch.force_edgecolor": True,
        "text.color": "dimgrey",
        "xtick.bottom": False,
        "xtick.color": "dimgrey",
        "xtick.direction": "out",
        "xtick.top": False,
        "ytick.color": "dimgrey",
        "ytick.direction": "out",
        "ytick.left": False,
        "ytick.right": False,
    },
)
sns.set_context(
    "notebook", rc={"font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14}
)

plt.legend(frameon=False)
sns.despine(left=True, bottom=True)

# #Loop through these labels
# for n, i in enumerate(labels):
#     #Create an axis text object
#     ax.text(X[n]-0.003, #X location of text (with adjustment)
#             n, #Y location
#             s=f'{round(X[n],3)}%', #Required label with formatting
#             va='center', #Vertical alignment
#             ha='right', #Horizontal alignment
#             color='white', #Font colour and size
#             fontsize=12)
