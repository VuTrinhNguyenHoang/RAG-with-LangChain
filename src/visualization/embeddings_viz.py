from sklearn.manifold import TSNE
import plotly.graph_objects as go
import numpy as np
import os
from typing import List, Dict, Any, Optional, Tuple


def create_embeddings_visualization(
    vectors: np.ndarray,
    doc_types: List[str],
    documents: List[str],
    random_state: int = 42
) -> go.Figure:
    """
    Create a TSNE visualization of embeddings
    
    Args:
        vectors: Array of embedding vectors
        doc_types: Document types corresponding to vectors
        documents: Document texts corresponding to vectors
        random_state: Random state for TSNE
        
    Returns:
        Plotly figure object
    """
    # Define color mapping
    color_map = {
        'company': 'blue',
        'employees': 'green',
        'visas': 'red',
        'schools': 'orange'
    }
    
    # Map document types to colors
    colors = [color_map.get(t, 'gray') for t in doc_types]
    
    # Create TSNE model
    tsne = TSNE(n_components=2, random_state=random_state)
    reduced_vectors = tsne.fit_transform(vectors)
    
    # Create Plotly figure
    fig = go.Figure(data=[go.Scatter(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        mode='markers',
        marker=dict(size=5, color=colors, opacity=0.8),
        text=[f"Type: {t}<br>Text: {d[:50]}..." for t, d in zip(doc_types, documents)],
        hoverinfo='text'
    )])
    
    fig.update_layout(
        title='2D Visualization of Vector Store Embeddings',
        scene=dict(xaxis_title='x', yaxis_title='y'),
        width=800,
        height=600,
        margin=dict(r=20, b=10, l=10, t=40)
    )
    
    return fig


def save_visualization(
    fig: go.Figure, 
    output_dir: str,
    filename: str = 'embeddings_visualization.html'
) -> str:
    """
    Save visualization to HTML file
    
    Args:
        fig: Plotly figure to save
        output_dir: Directory to save HTML file
        filename: Name of HTML file
        
    Returns:
        Path to saved HTML file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    fig.write_html(output_path)
    return output_path
