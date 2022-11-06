import argparse
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=str, default="")
    parser.add_argument('--f1', type=str, default="")
    parser.add_argument('--d', type=str, default="")
    parser.add_argument('--f2', type=str, default="")
    parser.add_argument('--t', type=str, default="")
    args = parser.parse_args()
    df = pd.read_csv(args.p + args.f1, args.d)


    # Box Plot visualization MSSubClass with Pandas
    bp = df.boxplot(figsize=(7, 7),
                        widths=0.6,
                        rot=0,
                        fontsize='7',
                        grid=False,
                        showmeans=True,
                        return_type='dict')
#    bp['boxes'][0].set(facecolor='lightgrey')
#    bp['boxes'][1].set(facecolor='white')


    for i, d in enumerate(df):
        y = df[d]
        x = np.random.normal(i + 1, 0.04, len(y))
        plt.scatter(x, y, c=['k'], s=15)

    medians = [round(item.get_ydata()[0], 1) for item in bp['medians']]
    means = [round(item.get_ydata()[0], 1) for item in bp['means']]
    print(f'Medians: {medians}\n'
          f'Means:   {means}')

    minimums = [round(item.get_ydata()[0], 1) for item in bp['caps']][::2]
    maximums = [round(item.get_ydata()[0], 1) for item in bp['caps']][1::2]
    print(f'Minimums: {minimums}\n'
          f'Maximums: {maximums}')

    q1 = [round(min(item.get_ydata()), 1) for item in bp['boxes']]
    q3 = [round(max(item.get_ydata()), 1) for item in bp['boxes']]
    print(f'Q1: {q1}\n'
          f'Q3: {q3}')
    fliers = [item.get_ydata() for item in bp['fliers']]
    lower_outliers = []
    upper_outliers = []
    for i in range(len(fliers)):
        lower_outliers_by_box = []
        upper_outliers_by_box = []
        for outlier in fliers[i]:
            if outlier < q1[i]:
                lower_outliers_by_box.append(round(outlier, 1))
            else:
                upper_outliers_by_box.append(round(outlier, 1))
        lower_outliers.append(lower_outliers_by_box)
        upper_outliers.append(upper_outliers_by_box)
    print(f'Lower outliers: {lower_outliers[1][0:3]}\n'
          f'Upper outliers: {upper_outliers[:][0:3]}')
    lower_outliers[0] = lower_outliers[0][0:4]
    lower_outliers[1] = lower_outliers[1][0:4]

    upper_outliers[0] = upper_outliers[0][0:3]
    upper_outliers[1] = upper_outliers[1][0:3]

    upper_outliers[0].sort(reverse=True)
    upper_outliers[1].sort(reverse=True)


    data = [medians, means, minimums, maximums, q1, q3, lower_outliers, upper_outliers]
    print(data)
    rows = ['medians', 'means', 'minimums', 'maximums', 'q1', 'q3', 'lower_outliers', 'upper_outliers']
    columns = df.columns
    # Get some pastel shades for the colors
   # colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
    n_rows = len(data)

    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4

    # Initialize the vertical-offset for the stacked bar chart.
    y_offset = np.zeros(len(columns))

    # Plot bars and create text labels for the table
    cell_text = []
    for row in range(n_rows):
        y_offset = y_offset + 1
        cell_text.append(['%1.1f' % (x / 1000.0) for x in y_offset])
    # Reverse colors and text labels to display the last value at the top.
    # colors = colors[::-1]
    cell_text.reverse()

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=data,
                          rowLabels=rows,
                          # rowColours=colors,
                          cellLoc='center',
                          colLabels=columns,
                          loc='bottom')
    the_table.set_fontsize(7)
    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.title(args.t)
    plt.savefig(args.p + args.f2+ "_analyze")
    #plt.show()
    plt.figure()
    plt.title(args.t)
    ax = sns.boxplot(
        data=df,
        showmeans=True,
        showfliers = False,
        color="lightgrey")

    ax = sns.swarmplot(data=df, color="k",size=3)
    max_maximums = max(maximums[0],maximums[1])
    min_a = float('inf')
    min_b = float('inf')
    if upper_outliers[0]:
        min_a = min(upper_outliers[0])
    else:
        min_a = maximums[0]+1

    if upper_outliers[1]:
        min_b = min(upper_outliers[1])
    else:
        min_b = maximums[1] + 1

    max_y = 15
    if max_y > min(min_a, min_b):
        max_y = min(min_a, min_b)

    if max_y < max_maximums:
        max_y = max_maximums +1

    #plt.yticks(np.arange(0,max_y, 1))
    plt.ylim(-1, max_y+max_y*10/100)

    #plt.xticks([])

    #plt.show()
    plt.savefig(args.p + args.f2 + "_plot")