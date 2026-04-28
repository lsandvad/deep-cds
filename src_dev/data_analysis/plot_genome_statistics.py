import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots

plt.style.use(['science', 'nature'])

# Load genome statistics data
df_statistics = pd.read_csv("../../data/processed_data/dataset_information/genomes_info_with_partitions.csv")
df_statistics = df_statistics.rename(columns={"Unnamed: 0":"accession"})

colors_dict = {
    'train': "#3BC552",   # Teal
    'val': '#4A90D9',     # Rose
    'test': '#C88B3A',    # Coral (more visible than sand)
    'hist': "#030050"     # Deep indigo
}


### Plot coding percentage statistics by partition ###

# Boxplot + strip (jittered points)
fig, ax = plt.subplots(figsize=(6, 5))

partition_order = ['train', 'val', 'test']
partition_labels = ['Train', 'Validation', 'Test']
positions = [1, 2, 3]

# Draw boxplots
data_by_partition = [df_statistics[df_statistics['partition'] == p]['coding_percentage'].values 
                     for p in partition_order]

bp = ax.boxplot(data_by_partition, positions=positions, widths=0.5, patch_artist=True,
                showfliers=False)

for i, (patch, partition) in enumerate(zip(bp['boxes'], partition_order)):
    patch.set_facecolor(colors_dict[partition])
    patch.set_alpha(0.6)

for median in bp['medians']:
    median.set_color('#332288')
    median.set_linewidth(1.5)

# Add jittered points
for i, (partition, pos) in enumerate(zip(partition_order, positions)):
    data = df_statistics[df_statistics['partition'] == partition]['coding_percentage']
    jitter = np.random.uniform(-0.15, 0.15, size=len(data))
    ax.scatter(pos + jitter, data, alpha=0.4, s=10, color=colors_dict[partition], edgecolor='none')

ax.set_xticks(positions)
ax.set_xticklabels(partition_labels)
ax.set_xlabel('Partition')
ax.set_ylabel('Coding Percentage')

plt.tight_layout()
#plt.savefig('coding_percentage_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# Summary statistics by partition
print("\nSummary Statistics (Coding Percentage) by Partition:")
print(df_statistics.groupby('partition')['coding_percentage'].describe())


### Plot genome length statistics by partition ###

# Boxplot + strip (jittered points)
fig, ax = plt.subplots(figsize=(6, 5))

partition_order = ['train', 'val', 'test']
partition_labels = ['Train', 'Validation', 'Test']
positions = [1, 2, 3]

# Draw boxplots
data_by_partition = [df_statistics[df_statistics['partition'] == p]['genome_length_kb'].values 
                     for p in partition_order]

bp = ax.boxplot(data_by_partition, positions=positions, widths=0.5, patch_artist=True,
                showfliers=False)

for i, (patch, partition) in enumerate(zip(bp['boxes'], partition_order)):
    patch.set_facecolor(colors_dict[partition])
    patch.set_alpha(0.6)

for median in bp['medians']:
    median.set_color('#332288')
    median.set_linewidth(1.5)

# Add jittered points
for i, (partition, pos) in enumerate(zip(partition_order, positions)):
    data = df_statistics[df_statistics['partition'] == partition]['genome_length_kb']
    jitter = np.random.uniform(-0.15, 0.15, size=len(data))
    ax.scatter(pos + jitter, data, alpha=0.4, s=10, color=colors_dict[partition], edgecolor='none')

ax.set_xticks(positions)
ax.set_xticklabels(partition_labels)
ax.set_xlabel('Partition')
ax.set_ylabel('Genome Length (kbp)')

plt.tight_layout()
#plt.savefig('genome_length_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# Summary statistics by partition
print("\nSummary Statistics (Genome Length) by Partition:")
print(df_statistics.groupby('partition')['genome_length_kb'].describe())


### Plot GC-content statistics by partition ###

# Boxplot + strip (jittered points)
fig, ax = plt.subplots(figsize=(6, 5))

partition_order = ['train', 'val', 'test']
partition_labels = ['Train', 'Validation', 'Test']
positions = [1, 2, 3]

# Draw boxplots
data_by_partition = [df_statistics[df_statistics['partition'] == p]['gc_content'].values 
                     for p in partition_order]

bp = ax.boxplot(data_by_partition, positions=positions, widths=0.5, patch_artist=True,
                showfliers=False)

for i, (patch, partition) in enumerate(zip(bp['boxes'], partition_order)):
    patch.set_facecolor(colors_dict[partition])
    patch.set_alpha(0.6)

for median in bp['medians']:
    median.set_color('#332288')
    median.set_linewidth(1.5)

# Add jittered points
for i, (partition, pos) in enumerate(zip(partition_order, positions)):
    data = df_statistics[df_statistics['partition'] == partition]['gc_content']
    jitter = np.random.uniform(-0.15, 0.15, size=len(data))
    ax.scatter(pos + jitter, data, alpha=0.4, s=10, color=colors_dict[partition], edgecolor='none')

ax.set_xticks(positions)
ax.set_xticklabels(partition_labels)
ax.set_xlabel('Partition')
ax.set_ylabel('GC-content')

plt.tight_layout()
#plt.savefig('gc_content_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# Summary statistics by partition
print("\nSummary Statistics (GC-content) by Partition:")
print(df_statistics.groupby('partition')['gc_content'].describe())


### Plot CDS median length statistics by partition ###

# Boxplot + strip (jittered points)
fig, ax = plt.subplots(figsize=(6, 5))

partition_order = ['train', 'val', 'test']
partition_labels = ['Train', 'Validation', 'Test']
positions = [1, 2, 3]

# Draw boxplots
data_by_partition = [df_statistics[df_statistics['partition'] == p]['cds_median_length'].values 
                     for p in partition_order]

bp = ax.boxplot(data_by_partition, positions=positions, widths=0.5, patch_artist=True,
                showfliers=False)

for i, (patch, partition) in enumerate(zip(bp['boxes'], partition_order)):
    patch.set_facecolor(colors_dict[partition])
    patch.set_alpha(0.6)

for median in bp['medians']:
    median.set_color('#332288')
    median.set_linewidth(1.5)

# Add jittered points
for i, (partition, pos) in enumerate(zip(partition_order, positions)):
    data = df_statistics[df_statistics['partition'] == partition]['cds_median_length']
    jitter = np.random.uniform(-0.15, 0.15, size=len(data))
    ax.scatter(pos + jitter, data, alpha=0.4, s=10, color=colors_dict[partition], edgecolor='none')

ax.set_xticks(positions)
ax.set_xticklabels(partition_labels)
ax.set_xlabel('Partition')
ax.set_ylabel('Median CDS Length (bp)')

plt.tight_layout()
#plt.savefig('median_cds_length_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# Summary statistics by partition
print("\nSummary Statistics (Median CDS Length) by Partition:")
print(df_statistics.groupby('partition')['cds_median_length'].describe())


# Combined figure with all 4 plots; Coding Percentage, Genome Length, GC-content, Median CDS Length
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

partition_order = ['train', 'val', 'test']
partition_labels = {'train': 'Train', 'val': 'Validation', 'test': 'Test'}
positions = [1, 2, 3]

# Data columns and labels for each subplot
plot_configs = [
    ('coding_percentage', r'Coding Percentage (\%)', 'A)'),
    ('genome_length_kb', 'Genome Length (kbp)', 'B)'),
    ('gc_content', r'GC-content (\%)', 'C)'),
    ('cds_median_length', 'Median CDS Length (bp)', 'D)')
]

# Create each subplot
for ax, (column, ylabel, panel_label) in zip(axes.flatten(), plot_configs):
    # Draw boxplots
    data_by_partition = [df_statistics[df_statistics['partition'] == p][column].values 
                         for p in partition_order]
    
    bp = ax.boxplot(data_by_partition, positions=positions, widths=0.5, patch_artist=True,
                    showfliers=False)
    
    for i, (patch, partition) in enumerate(zip(bp['boxes'], partition_order)):
        patch.set_facecolor(colors_dict[partition])
        patch.set_alpha(0.6)
    
    for median in bp['medians']:
        median.set_color('#332288')
        median.set_linewidth(1.5)
    
    # Add jittered points
    scatter_handles = []
    for i, (partition, pos) in enumerate(zip(partition_order, positions)):
        data = df_statistics[df_statistics['partition'] == partition][column]
        jitter = np.random.uniform(-0.15, 0.15, size=len(data))
        sc = ax.scatter(pos + jitter, data, alpha=0.4, s=15, color=colors_dict[partition], 
                       edgecolor='none', label=partition_labels[partition])
        scatter_handles.append(sc)
    
    # Remove x-axis labels (will use shared legend instead)
    ax.set_xticks([])
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis='y', labelsize=12)
    
    # Add panel label
    ax.text(0.02, 0.98, panel_label, transform=ax.transAxes, fontsize=16, 
            fontweight='bold', va='top')

# Create shared legend
handles = [plt.scatter([], [], color=colors_dict[p], s=60, alpha=0.6, label=partition_labels[p]) 
           for p in partition_order]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=14, 
           frameon=True, bbox_to_anchor=(0.5, 0.01))

plt.tight_layout()
plt.subplots_adjust(bottom=0.08)
plt.savefig('../../illustrations/genome_statistics/genome_statistics_combined.png', dpi=300, bbox_inches='tight')
plt.close()