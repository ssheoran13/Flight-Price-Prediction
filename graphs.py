import preprocess

df = preprocess.preprocess(graphs=True)
print(df.head())


# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
# data.iloc[:, 0].plot.kde(ax=ax1, title="Feature 0")
# plt.show()
