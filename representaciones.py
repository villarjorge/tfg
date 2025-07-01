# Contiene funciones para representar los resultados de la simulación
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D

def create_3d_ridge_plot(data, x_values=None, y_values=None, 
                        colors=None, alpha=0.8, cmap='viridis',
                        title="3D Ridge Plot", xlabel="X", ylabel="Y", zlabel="Z"):
    """
    Create a 3D ridge plot using PolyCollection.
    
    Parameters:
    - data: 2D array where each row is a curve to be plotted
    - x_values: x-axis values (if None, uses indices)
    - y_values: y-axis values for each curve (if None, uses indices)
    - colors: colors for each polygon
    - alpha: transparency
    - cmap: colormap name
    - title, xlabel, ylabel, zlabel: plot labels
    """
    
    # Convert to numpy array
    data = np.array(data)
    n_curves, n_points = data.shape
    
    # Create x values if not provided
    if x_values is None:
        x_values = np.arange(n_points)
    
    # Create y values if not provided  
    if y_values is None:
        y_values = np.arange(n_curves)
    
    # Create the 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create polygons for PolyCollection
    polygons = []
    
    for i in range(n_curves):
        # Create vertices for the polygon
        # Start at baseline, go along the curve, return to baseline
        verts = []
        
        # Start point (baseline)
        verts.append([x_values[0], 0])
        
        # Add all the curve points
        for j in range(n_points):
            verts.append([x_values[j], data[i, j]])
        
        # End point (baseline)
        verts.append([x_values[-1], 0])
        
        polygons.append(verts)
    
    # Create colors if not provided
    if colors is None:
        colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, n_curves))
    
    # Create PolyCollection
    poly_collection = PolyCollection(polygons, alpha=alpha, 
                                   facecolors=colors, edgecolors='black', linewidths=0.5)
    
    # Add collection to 3D axis
    ax.add_collection3d(poly_collection, zs=y_values, zdir='y')
    
    # Set axis limits
    ax.set_xlim(x_values[0], x_values[-1])
    ax.set_ylim(y_values[0], y_values[-1])
    ax.set_zlim(0, np.max(data) * 1.1)
    
    # Set labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    
    return fig, ax

def create_advanced_3d_ridge(data, x_values=None, y_values=None, 
                            colors=None, alpha=0.8, shade=True,
                            view_angle=(30, 45)):
    """
    Advanced 3D ridge plot with better shading and perspective.
    """
    
    data = np.array(data)
    n_curves, n_points = data.shape
    
    if x_values is None:
        x_values = np.arange(n_points)
    if y_values is None:
        y_values = np.arange(n_curves)
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create polygons with better vertex handling
    polygons = []
    
    for i in range(n_curves):
        # Create closed polygon vertices
        verts = []
        
        # Bottom edge (y=0)
        for j in range(n_points):
            verts.append([x_values[j], 0])
        
        # Top edge (actual data, reversed for proper polygon)
        for j in range(n_points-1, -1, -1):
            verts.append([x_values[j], data[i, j]])
        
        polygons.append(verts)
    
    # Create color mapping
    if colors is None:
        # Use height-based coloring
        max_heights = np.max(data, axis=1)
        norm_heights = (max_heights - np.min(max_heights)) / (np.max(max_heights) - np.min(max_heights))
        colors = plt.cm.plasma(norm_heights)
    
    # Create PolyCollection with enhanced properties
    poly_collection = PolyCollection(polygons, 
                                   alpha=alpha,
                                   facecolors=colors,
                                   edgecolors='black',
                                   linewidths=0.3)
    
    # Add to 3D plot
    ax.add_collection3d(poly_collection, zs=y_values, zdir='y')
    
    # Enhanced axis setup
    ax.set_xlim(x_values[0], x_values[-1])
    ax.set_ylim(y_values[0], y_values[-1])
    ax.set_zlim(0, np.max(data) * 1.1)
    
    # Set viewing angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # Styling
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('3D Ridge Plot with PolyCollection', fontsize=14, pad=20)
    
    # Remove grid for cleaner look (optional)
    ax.grid(False)
    
    return fig, ax

# Alternative approach using surface plot for comparison
def create_surface_ridge(data, x_values=None, y_values=None, cmap='viridis'):
    """
    Create ridge plot using surface plot for comparison.
    """
    data = np.array(data)
    n_curves, n_points = data.shape
    
    if x_values is None:
        x_values = np.arange(n_points)
    if y_values is None:
        y_values = np.arange(n_curves)
    
    # Create meshgrid
    X, Y = np.meshgrid(x_values, y_values)
    Z = data
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create surface plot
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.8,
                          linewidth=0.5, edgecolors='black')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.set_title('Surface Ridge Plot')
    
    # Add colorbar
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    return fig, ax

def create_ridge_plot(data, x_values=None, offset=1.0, fill=False, colors=None, 
                     labels=None, title="Ridge Plot", xlabel="X", ylabel="Series"):
    """
    Create a ridge/cascade plot with stacked time series.
    
    Parameters:
    - data: 2D array or list of arrays, each row/array is a time series
    - x_values: x-axis values (if None, uses indices)
    - offset: vertical offset between series
    - fill: whether to fill under curves
    - colors: list of colors for each series
    - labels: list of labels for each series
    - title: plot title
    - xlabel, ylabel: axis labels
    """
    
    # Convert data to numpy array if needed
    if isinstance(data, list):
        # Handle lists of different lengths
        max_len = max(len(series) for series in data)
        data_array = np.full((len(data), max_len), np.nan)
        for i, series in enumerate(data):
            data_array[i, :len(series)] = series
        data = data_array
    
    n_series, n_points = data.shape
    
    # Create x values if not provided
    if x_values is None:
        x_values = np.arange(n_points)
    
    # Create colors if not provided
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, n_series))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each series with vertical offset
    for i in range(n_series):
        y_offset = i * offset
        y_data = data[i] + y_offset
        
        # Remove NaN values for plotting
        valid_mask = ~np.isnan(y_data)
        x_valid = x_values[valid_mask]
        y_valid = y_data[valid_mask]
        
        if fill:
            # Fill under the curve
            ax.fill_between(x_valid, y_offset, y_valid, 
                          alpha=0.3, color=colors[i])
        
        # Plot the line
        label = labels[i] if labels and i < len(labels) else f'Series {i+1}'
        ax.plot(x_valid, y_valid, color=colors[i], linewidth=1.5, label=label)
    
    # Add horizontal reference lines
    for i in range(n_series):
        ax.axhline(y=i * offset, color='gray', alpha=0.3, linewidth=0.5, linestyle='--')
    
    # Customize the plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Set y-axis labels to show series indices
    y_ticks = np.arange(n_series) * offset
    y_tick_labels = [f'{i}' for i in range(n_series)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    
    plt.tight_layout()
    return fig, ax

# Example usage
if __name__ == "__main__":
    
    # Example 1: Recreate something similar to your image
    print("Creating ridge plot similar to your example...")
    
    # Generate sample data similar to the image
    n_series = 45
    n_points = 200
    x = np.linspace(0, 200, n_points)
    
    # Create data with varying patterns
    data = []
    for i in range(n_series):
        # Create base signal with different frequencies and phases
        freq1 = 0.05 + 0.02 * np.random.random()
        freq2 = 0.1 + 0.05 * np.random.random()
        phase1 = 2 * np.pi * np.random.random()
        phase2 = 2 * np.pi * np.random.random()
        
        # Combine sine waves with noise
        signal = (2 * np.sin(2 * np.pi * freq1 * x + phase1) + 
                 1.5 * np.sin(2 * np.pi * freq2 * x + phase2) + 
                 0.5 * np.random.normal(0, 1, n_points))
        
        data.append(signal)
    
    # Create the ridge plot
    fig1, ax1 = create_ridge_plot(data, x_values=x, offset=1.0, 
                                 title="Ridge Plot - Time Series Stack",
                                 xlabel="Space [spatial index]", 
                                 ylabel="Time [frame number]")
    
    # Invert y-axis to match your image (highest index at top)
    ax1.invert_yaxis()
    
    plt.show()
    
    # Example 2: Simpler version with fewer series
    print("\nCreating simpler ridge plot...")
    
    # Generate sample data
    x2 = np.linspace(0, 10, 100)
    data2 = []
    
    for i in range(8):
        # Different wave patterns
        wave = np.sin(x2 + i * 0.5) * np.exp(-x2/10) + 0.2 * np.random.normal(0, 1, 100)
        data2.append(wave)
    
    fig2, ax2 = create_ridge_plot(data2, x_values=x2, offset=2.0, fill=True,
                                 title="Ridge Plot with Fill", 
                                 xlabel="Time", ylabel="Channel")
    plt.show()
    
    # Example 3: Manual approach for more control
    print("\nManual ridge plot approach...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sample data
    x3 = np.linspace(0, 4*np.pi, 200)
    n_curves = 10
    offset = 1.5
    
    for i in range(n_curves):
        # Create different wave patterns
        y = np.sin(x3 + i * 0.3) * np.exp(-x3/10) + i * offset
        
        # Add some noise
        y += 0.1 * np.random.normal(0, 1, len(x3))
        
        # Plot the curve
        ax.plot(x3, y, 'k-', linewidth=1, alpha=0.8)
        
        # Optional: fill under curve
        ax.fill_between(x3, i * offset, y, alpha=0.2, color=f'C{i%10}')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Offset Series')
    ax.set_title('Manual Ridge Plot')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Example 4: Using actual time series data format
    print("\nRidge plot with time series data...")
    
    # Simulate spectral data or similar
    frequencies = np.linspace(0, 50, 200)
    time_points = 20
    
    # Create data matrix (time_points x frequencies)
    spectral_data = np.zeros((time_points, len(frequencies)))
    
    for t in range(time_points):
        # Simulate evolving spectral peaks
        peak1 = 10 + 2 * np.sin(t * 0.3)
        peak2 = 30 + 3 * np.cos(t * 0.2)
        
        spectral_data[t] = (np.exp(-0.5 * ((frequencies - peak1) / 2)**2) + 
                           0.7 * np.exp(-0.5 * ((frequencies - peak2) / 3)**2) +
                           0.1 * np.random.normal(0, 1, len(frequencies)))
    
    fig4, ax4 = create_ridge_plot(spectral_data, x_values=frequencies, 
                                 offset=0.5, fill=True,
                                 title="Spectral Evolution Over Time",
                                 xlabel="Frequency (Hz)", 
                                 ylabel="Time Point")
    
    # Invert y-axis so newest data is on top
    ax4.invert_yaxis()
    plt.show()

    # Generate sample data similar to your image
    print("Creating 3D ridge plot with PolyCollection...")
    
    # Parameters
    n_curves = 25
    n_points = 200
    x = np.linspace(0, 200, n_points)
    y_positions = np.linspace(5, 25, n_curves)
    
    # Generate data with a main peak that varies
    data = []
    for i, y_pos in enumerate(y_positions):
        # Create a main gaussian peak with some variation
        center = 100 + 20 * np.sin(i * 0.3)  # Moving peak center
        width = 25 + 10 * np.cos(i * 0.2)    # Varying width
        height = 2.5 + 0.5 * np.sin(i * 0.4) # Varying height
        
        # Main gaussian
        curve = height * np.exp(-0.5 * ((x - center) / width)**2)
        
        # Add some smaller peaks
        curve += 0.3 * np.exp(-0.5 * ((x - 50) / 15)**2)
        curve += 0.2 * np.exp(-0.5 * ((x - 170) / 12)**2)
        
        # Add some noise
        curve += 0.05 * np.random.normal(0, 1, len(x))
        
        # Ensure non-negative
        curve = np.maximum(curve, 0)
        
        data.append(curve)
    
    data = np.array(data)
    
    # Create the 3D ridge plot
    fig1, ax1 = create_advanced_3d_ridge(data, x_values=x, y_values=y_positions,
                                        alpha=0.7, view_angle=(25, 45))
    plt.show()
    
    # Example 2: Simpler case
    print("\nCreating simpler 3D ridge plot...")
    
    # Simple test data
    x2 = np.linspace(0, 10, 100)
    data2 = []
    
    for i in range(15):
        # Create different gaussian peaks
        center = 5 + 2 * np.sin(i * 0.5)
        curve = 2 * np.exp(-0.5 * ((x2 - center) / 1.5)**2) + 0.2
        data2.append(curve)
    
    fig2, ax2 = create_3d_ridge_plot(data2, x_values=x2, 
                                    title="Simple 3D Ridge Plot",
                                    xlabel="Position", ylabel="Time", zlabel="Amplitude")
    plt.show()
    
    # Example 3: Compare with surface plot
    print("\nComparing with surface plot...")
    
    fig3, ax3 = create_surface_ridge(data2, x_values=x2, cmap='plasma')
    plt.show()
    
    # Example 4: Custom colors
    print("\nCreating plot with custom colors...")
    
    # Create custom color array
    custom_colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(data2)))
    
    fig4, ax4 = create_3d_ridge_plot(data2, x_values=x2, colors=custom_colors,
                                    alpha=0.8, title="Custom Colored 3D Ridge Plot")
    plt.show()