import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
 
FEATURE_NAMES = ['sepal length', 'sepal width', 'petal length', 'petal width']
CLASS_NAMES   = ['setosa', 'versicolor', 'virginica']
 
NODE_STYLE = dict(boxstyle='round,pad=0.4', facecolor='#E1F5EE', edgecolor='#0F6E56', linewidth=0.8)
LEAF_COLORS = ['#EEEDFE', '#FAEEDA', '#FAECE7']
LEAF_EDGES  = ['#534AB7', '#854F0B', '#993C1D']
REG_LEAF_STYLE = dict(boxstyle='round,pad=0.4', facecolor='#FCE9E9', edgecolor='#B24242', linewidth=0.8)
 
 
def _tree_width(node):
    if not isinstance(node, dict):
        return 1
    return _tree_width(node['left']) + _tree_width(node['right'])


def _feature_name(feature_names, index):
    if feature_names and 0 <= index < len(feature_names):
        return feature_names[index]
    return f"x{index}"
 
 
def _draw_node(ax, node, x, y, dx, dy, parent_xy=None):
    is_leaf = not isinstance(node, dict)
 
    if is_leaf:
        cls   = int(round(node))
        label = f"{CLASS_NAMES[cls]}\n(klasa {cls})"
        bbox  = dict(boxstyle='round,pad=0.4',
                     facecolor=LEAF_COLORS[cls % len(LEAF_COLORS)],
                     edgecolor=LEAF_EDGES[cls % len(LEAF_EDGES)],
                     linewidth=0.8)
    else:
        feat  = FEATURE_NAMES[node['spInd']]
        label = f"{feat}\n<= {node['spVal']:.2f}"
        bbox  = NODE_STYLE
 
    ax.text(x, y, label, ha='center', va='center', fontsize=7.5,
            bbox=bbox, zorder=3)
 
    if parent_xy:
        ax.annotate('', xy=(x, y + 0.022), xytext=parent_xy,
                    arrowprops=dict(arrowstyle='->', color='#888780', lw=0.8))
 
    if is_leaf:
        return
 
    lw = _tree_width(node['left'])
    rw = _tree_width(node['right'])
    total = lw + rw
 
    x_left  = x - dx * rw / total
    x_right = x + dx * lw / total
    ny = y - dy
 
    ax.text((x + x_right) / 2, y - dy * 0.35, '>',
            ha='center', va='center', fontsize=7, color='#5F5E5A')
    ax.text((x + x_left) / 2,  y - dy * 0.35, '<=',
            ha='center', va='center', fontsize=7, color='#5F5E5A')
 
    _draw_node(ax, node['left'],  x_left,  ny, dx * lw / total,  dy, (x, y - 0.022))
    _draw_node(ax, node['right'], x_right, ny, dx * rw / total, dy, (x, y - 0.022))


def _draw_reg_node(ax, node, x, y, dx, dy, feature_names, parent_xy=None):
    is_leaf = not isinstance(node, dict)

    if is_leaf:
        label = f"y = {float(node):.2f}"
        bbox = REG_LEAF_STYLE
    else:
        feat = _feature_name(feature_names, node['spInd'])
        label = f"{feat}\n<= {node['spVal']:.3f}"
        bbox = NODE_STYLE

    ax.text(x, y, label, ha='center', va='center', fontsize=7.5, bbox=bbox, zorder=3)

    if parent_xy:
        ax.annotate('', xy=(x, y + 0.022), xytext=parent_xy,
                    arrowprops=dict(arrowstyle='->', color='#888780', lw=0.8))

    if is_leaf:
        return

    lw = _tree_width(node['left'])
    rw = _tree_width(node['right'])
    total = lw + rw

    x_left = x - dx * rw / total
    x_right = x + dx * lw / total
    ny = y - dy

    ax.text((x + x_right) / 2, y - dy * 0.35, '>',
            ha='center', va='center', fontsize=7, color='#5F5E5A')
    ax.text((x + x_left) / 2, y - dy * 0.35, '<=',
            ha='center', va='center', fontsize=7, color='#5F5E5A')

    _draw_reg_node(ax, node['left'], x_left, ny, dx * lw / total, dy, feature_names, (x, y - 0.022))
    _draw_reg_node(ax, node['right'], x_right, ny, dx * rw / total, dy, feature_names, (x, y - 0.022))
 
 
def plot_tree(tree, title='Drzewo decyzyjne – Iris'):
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 0.15)
    ax.axis('off')
    fig.patch.set_facecolor('#FAFAF8')
 
    _draw_node(ax, tree, x=0, y=0, dx=0.9, dy=0.22)
 
    legend_patches = [
        mpatches.Patch(facecolor='#E1F5EE', edgecolor='#0F6E56', label='Węzeł wewnętrzny'),
        mpatches.Patch(facecolor=LEAF_COLORS[0], edgecolor=LEAF_EDGES[0], label='setosa'),
        mpatches.Patch(facecolor=LEAF_COLORS[1], edgecolor=LEAF_EDGES[1], label='versicolor'),
        mpatches.Patch(facecolor=LEAF_COLORS[2], edgecolor=LEAF_EDGES[2], label='virginica'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=8, framealpha=0.8)
    ax.set_title(title, fontsize=11, pad=8)
 
    plt.tight_layout()
    plt.show()


def plot_tree_regression(tree, feature_names, title='Drzewo regresyjne'):
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 0.15)
    ax.axis('off')
    fig.patch.set_facecolor('#FAFAF8')

    _draw_reg_node(ax, tree, x=0, y=0, dx=0.9, dy=0.22, feature_names=feature_names)

    legend_patches = [
        mpatches.Patch(facecolor='#E1F5EE', edgecolor='#0F6E56', label='Węzeł podziału'),
        mpatches.Patch(facecolor=REG_LEAF_STYLE['facecolor'], edgecolor=REG_LEAF_STYLE['edgecolor'], label='Liść: wartość y'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=8, framealpha=0.8)
    ax.set_title(title, fontsize=11, pad=8)

    plt.tight_layout()
    plt.show()