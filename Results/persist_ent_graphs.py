import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu
import os
from matplotlib.lines import Line2D
from statannotations.Annotator import Annotator

# Plot settings for perssistent entropy results
path = './persistent results'
maps = ['Brainnetome',  'Gordon', 'Glasser']
condition = ['Young Adult','Aging']

data = {'Entropy 0': [],'Entropy 1': [], 'Entropy 2': [], 'Codes': [],'Atlas': [],'Group': []}
for c in range(len(condition)):
    for m in range(len(maps)):
       filename=os.path.join(path, condition[c] + '_' + maps[m] + '.txt')
       with open(filename) as f:
        lines = f.readlines()
        
        for line in lines:
            columns = line.strip().split(',') 
            data['Codes'].append((columns[0]))
            data['Entropy 0'].append(float(columns[1]))
            data['Entropy 1'].append(float(columns[2]))
            data['Entropy 2'].append(float(columns[3]))
            data['Atlas'].append(maps[m])
            data['Group'].append(condition[c])
    
df = pd.DataFrame(data)

fig, axes = plt.subplots(3, 1, figsize=(6,7), sharex=True)

palette = ['darkorchid', 'forestgreen']

sns.boxplot(data = df, x='Atlas', y='Entropy 0', hue='Group', ax = axes[0], palette = ['darkorchid','forestgreen'], showfliers = False, width = 0.5)
axes[0].set_title('Entropy 0')
axes[0].legend(title=None, fontsize = 10)
axes[0].set_ylim(4.45, 5)

sns.boxplot(data = df, x='Atlas', y='Entropy 1', hue='Group', ax = axes[1], palette = ['darkorchid','forestgreen'], showfliers = False, width = 0.5)
axes[1].set_title('Entropy 1')
axes[1].legend_.remove()
axes[1].set_ylim(2.5, 5)

sns.boxplot(data = df, x='Atlas', y='Entropy 2', hue='Group', ax = axes[2], palette = ['darkorchid','forestgreen'], showfliers = False, width = 0.5)
axes[2].set_title('Entropy 2')
axes[2].legend_.remove()
axes[2].set_ylim(0.5, 4.25)


for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(direction='in')
    ax.set_xticklabels([])
    ax.set_xlabel('')
    ax.set_title('')

axes[0].set_ylabel('Entropy 0th Homology', weight='bold', fontsize=10)
axes[1].set_ylabel('Entropy 1st Homology', weight='bold', fontsize=10)
axes[2].set_ylabel('Entropy 2nd Homology', weight='bold', fontsize=10)

atlas_names = ['Brainnetome', 'Gordon', 'Glasser']  
x_positions = [0, 1, 2]  

for x_pos, atlas_name in zip(x_positions, atlas_names):
    axes[2].text(x_pos, axes[2].get_ylim()[0] - 0.5,  
                 atlas_name, 
                 ha='center', va='center', fontsize=9)

axes[2].set_xlabel('Atlas', fontsize = 10, weight = 'bold', labelpad=15)

axes[2].set_xticks([0, 1, 2]) 
axes[2].set_xticklabels(['YA       A', 'YA       A', 'YA       A'], fontsize=9)

subplot_labels = ['(a)', '(b)', '(c)']  

for ax, label in zip(axes, subplot_labels):
    ax.text(-0.15, 1.05,  
            label, 
            transform=ax.transAxes,  
            ha='left', va='top', fontsize=10, fontweight='bold')

plt.tight_layout()

pair = [(('Brainnetome','Aging'),('Brainnetome','Young Adult')),
        (('Gordon','Aging'),('Gordon','Young Adult')),
        (('Glasser','Aging'),('Glasser','Young Adult'))]

annotator0 = Annotator(axes[0], pair, data=df, x='Atlas', y='Entropy 0', hue='Group',
                      order=['Brainnetome','Gordon','Glasser'])
annotator0.configure(test='Mann-Whitney', text_format='star', pvalue_thresholds=[[0.01, "**"],[0.05, "*"],[1, "ns"]])
annotator0.apply_and_annotate()

annotator1 = Annotator(axes[1], pair, data=df, x='Atlas', y='Entropy 1', hue='Group',
                      order=['Brainnetome','Gordon','Glasser'])
annotator1.configure(test='Mann-Whitney', text_format='star', pvalue_thresholds=[[0.01, "**"],[0.05, "*"],[1, "ns"]])
annotator1.apply_and_annotate()

annotator2 = Annotator(axes[2], pair, data=df, x='Atlas', y='Entropy 2', hue='Group',
                      order=['Brainnetome','Gordon','Glasser'])
annotator2.configure(test='Mann-Whitney', text_format='star', pvalue_thresholds=[[0.01, "**"],[0.05, "*"],[1, "ns"]])
annotator2.apply_and_annotate()

adjustments_y = {
    'Entropy 0': {'Brainnetome': -0.2, 'Gordon': -0.15, 'Glasser': -0.15},
    'Entropy 1': {'Brainnetome': -0.45, 'Gordon': -0.5, 'Glasser': -0.5},
    'Entropy 2': {'Brainnetome': -0.77, 'Gordon': -0.8, 'Glasser': -0.8}
}

adjustments_x = {
    'Entropy 0': {'Brainnetome': {'Young Adult': -0.15, 'Aging': 0.15},
                  'Gordon': {'Young Adult': -0.15, 'Aging': 0.15},
                  'Glasser': {'Young Adult': -0.15, 'Aging': 0.15}},
    'Entropy 1': {'Brainnetome': {'Young Adult': -0.15, 'Aging': 0.15},
                  'Gordon': {'Young Adult': -0.15, 'Aging': 0.15},
                  'Glasser': {'Young Adult': -0.15, 'Aging': 0.15}},
    'Entropy 2': {'Brainnetome': {'Young Adult': -0.15, 'Aging': 0.15},
                  'Gordon': {'Young Adult': -0.15, 'Aging': 0.15},
                  'Glasser': {'Young Adult': -0.15, 'Aging': 0.15}}
}
for ax, entropy in zip(axes, ['Entropy 0', 'Entropy 1', 'Entropy 2']):
    for i, atlas in enumerate(['Brainnetome', 'Gordon', 'Glasser']):
        for group in ['Young Adult', 'Aging']:

            group_data = df[(df['Atlas'] == atlas) & (df['Group'] == group)][entropy]

            mean_value = group_data.mean()
            std_value = group_data.std()

            formatted_text = f"{mean_value:.2f}({std_value*10:.0f})"

            position_y = mean_value - std_value + adjustments_y[entropy][atlas]

            position_x = i + adjustments_x[entropy][atlas][group]
            
            ax.text(position_x, position_y, formatted_text, 
                    ha='center', va='top', fontsize=8, color='black')

plt.tight_layout()
plt.show()

# Plot settings for sexes in persistent entropy results
path = "./persistent results"
atlas = ['fsaverage.BN_Atlas.32k_fs_LR_246regions', 'GlasserFreesurfer', 'Gordon333.32k_fs_LR_Tian_Subcortex_S1_3T']
condition = ['HCPYoungAdult', 'HCPAging']
names = ['Codes', 'Entropy 0', 'Entropy 1', 'Entropy 2']

young_bn = pd.read_csv(f"{path}/{condition[0]}_{atlas[0]}.txt", names=names)
young_glasser = pd.read_csv(f"{path}/{condition[0]}_{atlas[1]}.txt", names=names)
young_gordon = pd.read_csv(f"{path}/{condition[0]}_{atlas[2]}.txt", names=names)
aging_bn = pd.read_csv(f"{path}/{condition[1]}_{atlas[0]}.txt", names=names)
aging_glasser = pd.read_csv(f"{path}/{condition[1]}_{atlas[1]}.txt", names=names)
aging_gordon = pd.read_csv(f"{path}/{condition[1]}_{atlas[2]}.txt", names=names)

# Table containing additional sex data
young_tab = pd.read_excel('./HCPYoungAdult_data.xlsx')
young_tab = young_tab.rename(columns={'codes': 'Codes'})
aging_tab = pd.read_csv('./HCP_Aging_all_subjects.csv')
aging_tab = aging_tab.drop(0, axis=0)
aging_tab = aging_tab.rename(columns={'src_subject_id': 'Codes','sex': 'Sex','interview_age': 'Age'})
aging_tab['Age'] = aging_tab['Age'].map(lambda x: int(x)//12)

aging_bn = aging_bn.merge(aging_tab, on='Codes', how='inner')
aging_bn = aging_bn.dropna(axis=1)
aging_gordon = aging_gordon.merge(aging_tab, on = 'Codes', how = 'inner')
aging_gordon = aging_gordon.dropna(axis = 1)
aging_glasser = aging_glasser.merge(aging_tab, on = 'Codes', how = 'inner')
aging_glasser = aging_glasser.dropna(axis = 1)

young_bn = young_bn.merge(young_tab, on = 'Codes', how = 'inner')
young_bn = young_bn.dropna(axis = 1)
young_gordon = young_gordon.merge(young_tab, on = 'Codes', how = 'inner')
young_gordon = young_gordon.dropna(axis = 1)
young_glasser = young_glasser.merge(young_tab, on = 'Codes', how = 'inner')
young_glasser = young_glasser.dropna(axis = 1)

young_bn_M = young_bn[young_bn['Sex'] == 'M']
young_bn_F = young_bn[young_bn['Sex'] == 'F']
young_gordon_M = young_gordon[young_gordon['Sex'] == 'M']
young_gordon_F = young_gordon[young_gordon['Sex'] == 'F']
young_glasser_M = young_glasser[young_glasser['Sex'] == 'M']
young_glasser_F = young_glasser[young_glasser['Sex'] == 'F']

aging_bn_M = aging_bn[aging_bn['Sex'] == 'M']
aging_bn_F = aging_bn[aging_bn['Sex'] == 'F']
aging_gordon_M = aging_gordon[aging_gordon['Sex'] == 'M']
aging_gordon_F = aging_gordon[aging_gordon['Sex'] == 'F']
aging_glasser_M = aging_glasser[aging_glasser['Sex'] == 'M']
aging_glasser_F = aging_glasser[aging_glasser['Sex'] == 'F']

# Plotting persistent entropy results for Glasser atlas
fig, axes = plt.subplots(3, 1, figsize=(6,8), sharex=True)

young_glasser['Group'] = 'Young'
aging_glasser['Group'] = 'Aging'

combined_data = pd.concat([young_glasser, aging_glasser])

palette = {'Young F': 'royalblue', 'Young M': 'firebrick', 'Aging F': 'cornflowerblue', 'Aging M': 'lightcoral'}

combined_data['GroupSex'] = combined_data['Group'] + " " + combined_data['Sex']
sns.boxplot(data=combined_data, x='GroupSex', y='Entropy 0', ax=axes[0], palette=palette, order=['Young F', 'Young M', 'Aging F', 'Aging M'], showfliers=False, width=0.5)
axes[0].set_title('Entropy 0')
axes[0].set_ylim(5.15 , 6.5)
plt.title('Glasser', fontsize=12)

sns.boxplot(data=combined_data, x='GroupSex', y='Entropy 1', ax=axes[1], palette=palette, order=['Young F', 'Young M', 'Aging F', 'Aging M'], showfliers=False, width=0.5)
axes[1].set_title('Entropy 1')
axes[1].set_ylim(3, 5)

sns.boxplot(data=combined_data, x='GroupSex', y='Entropy 2', ax=axes[2], palette=palette, order=['Young F', 'Young M', 'Aging F', 'Aging M'], showfliers=False, width=0.5)
axes[2].set_title('Entropy 2')
axes[2].set_ylim(1, 5)


legend_elements = [
    Line2D([0], [0], color='royalblue', lw=2, label='Female - Young Adult'),
    Line2D([0], [0], color='firebrick', lw=2, label='Male -  Young Adult'),
    Line2D([0], [0], color='cornflowerblue', lw=2, label='Female - Aging'),
    Line2D([0], [0], color='lightcoral', lw=2, label='Male - Aging')
]
axes[0].legend(title='', handles=legend_elements, prop={'size': 8}, title_fontsize=8)

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Colocar os ticks para dentro
    ax.tick_params(direction='in')
    ax.set_xticklabels([])
    ax.set_xlabel('')
    ax.set_title('')

axes[0].set_ylabel('Entropy 0th Homology', weight='bold', fontsize=9)
axes[1].set_ylabel('Entropy 1st Homology', weight='bold', fontsize=9)
axes[2].set_ylabel('Entropy 2nd Homology', weight='bold', fontsize=9)
 

pair = [('Young F','Young M'),('Young F', 'Aging F'),('Young M','Aging M'),('Aging F','Aging M')]
annotator0 = Annotator(axes[0], pair, data=combined_data, x='GroupSex', y='Entropy 0', order=['Young F', 'Young M', 'Aging F', 'Aging M'])
annotator0.configure(test='Mann-Whitney', text_format='star', pvalue_thresholds=[[0.01, "**"],[0.05, "*"],[1, "ns"]])
annotator0.apply_and_annotate()

annotator1 = Annotator(axes[1], pair, data=combined_data, x='GroupSex', y='Entropy 1', order=['Young F', 'Young M', 'Aging F', 'Aging M'])
annotator1.configure(test='Mann-Whitney', text_format='star', pvalue_thresholds=[[0.01, "**"],[0.05, "*"],[1, "ns"]])
annotator1.apply_and_annotate()

annotator2 = Annotator(axes[2], pair, data=combined_data, x='GroupSex', y='Entropy 2', order=['Young F', 'Young M', 'Aging F', 'Aging M'])
annotator2.configure(test='Mann-Whitney', text_format='star', pvalue_thresholds=[[0.01, "**"],[0.05, "*"],[1, "ns"]])
annotator2.apply_and_annotate()

axes[2].set_xticks([0, 1, 2, 3]) 
axes[2].set_xticklabels(['F - YA', 'M - YA', 'F - A', 'M - A'], fontsize=9)

subplot_labels = ['(a)', '(b)', '(c)']

adjustments_x = {'Young F': 0, 'Young M': 0, 'Aging F': 0, 'Aging M': 0}

group_order = ['Young F', 'Young M', 'Aging F', 'Aging M']

for ax, entropy in zip(axes, ['Entropy 0', 'Entropy 1', 'Entropy 2']):
    xticks = ax.get_xticks()  

    y_min = ax.get_ylim()[0]
    offset = 0.07 * (ax.get_ylim()[1] - y_min)  

    for i, groupSex in enumerate(group_order):

        group_data = combined_data[combined_data['GroupSex'] == groupSex][entropy]

        mean_value = group_data.mean()
        std_value = group_data.std()

        formatted_text = f"{mean_value:.2f} ({std_value * 10:.0f})"

        position_x = xticks[i] + adjustments_x[groupSex]
        position_y = y_min + offset 

        ax.text(position_x, position_y, formatted_text,
                ha='center', va='top', fontsize=7, color='black')

plt.tight_layout()
plt.show()
