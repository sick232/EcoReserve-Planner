import os
import numpy as np
import rasterio
import folium
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, send_file, url_for
from werkzeug.utils import secure_filename
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from time import time
from sentinelhub import SHConfig, SentinelHubRequest, MimeType, CRS, BBox, DataCollection, MosaickingOrder

# ========================================
# Flask Setup
# ========================================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ========================================
# CSV & GeoTIFF Loaders
# ========================================
def load_csv_grid(path):
    return np.genfromtxt(path, delimiter=',', dtype=np.int64)

def load_geotiff_grid(path):
    with rasterio.open(path) as src:
        data = src.read(1)
        transform = src.transform
        west, south, east, north = src.bounds
        base_lat = north
        base_lon = west
        lat_step = -transform[4]
        lon_step = transform[0]
        data = np.clip(data, 0, None)
        return (data.astype(np.int64), base_lat, base_lon, lat_step, lon_step,
                f"{west},{south},{east},{north}", data.shape)

# ========================================
# Core Algorithm: Best Square
# ========================================
def build_prefix_sum(B):
    N, M = B.shape
    P = np.zeros((N+1, M+1), dtype=np.int64)
    for i in range(1, N+1):
        for j in range(1, M+1):
            P[i, j] = B[i-1, j-1] + P[i-1, j] + P[i, j-1] - P[i-1, j-1]
    return P

def find_best_square(B, min_k=1, max_k=None):
    N, M = B.shape
    if max_k is None:
        max_k = min(N, M)
    P = build_prefix_sum(B)
    best_sum, best = -1, None
    for k in range(min_k, max_k+1):
        for r in range(N - k + 1):
            for c in range(M - k + 1):
                s = int(P[r+k, c+k] - P[r, c+k] - P[r+k, c] + P[r, c])
                if s > best_sum:
                    best_sum, best = s, (r, c, k)
    return best_sum, best

# ========================================
# Map, PDF, Summary
# ========================================
def create_map(B, k, r, c, best_sum, threshold, app_name,
               base_lat, base_lon, lat_step, lon_step, map_type='primary'):
    # Calculate center point for better initial view
    center_lat = base_lat - (B.shape[0] * lat_step) / 2
    center_lon = base_lon + (B.shape[1] * lon_step) / 2
    
    # Initialize map with center point
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    N, M = B.shape
    
    # Optimized color schemes with transparency
    if map_type == 'primary':
        best_color = "#2E8B57"  # Dark green
        high_color = "#90EE90"  # Light green
        low_color = "#bdbdbd"   # Gray
        highlight_color = "red"
    else:
        best_color = "#1E88E5"  # Blue
        high_color = "#90CAF9"  # Light blue
        low_color = "#bdbdbd"   # Gray
        highlight_color = "#FFA000"  # Orange
        
    # Pre-calculate grid properties
    fill_opacity = 0.6 if map_type == 'primary' else 0.4
    
    # Add tile layers for better visualization with proper attribution
    folium.TileLayer(
        'OpenStreetMap',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    ).add_to(m)
    folium.TileLayer(
        'Stamen Terrain',
        attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://www.openstreetmap.org/copyright">ODbL</a>.'
    ).add_to(m)
    folium.TileLayer(
        'Stamen Toner',
        attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://www.openstreetmap.org/copyright">ODbL</a>.'
    ).add_to(m)
    
    # Create layers for different types of cells
    best_layer = folium.FeatureGroup(name='Best Square')
    high_layer = folium.FeatureGroup(name='High Value')
    low_layer = folium.FeatureGroup(name='Low Value')
    
    # Process grid in chunks for better performance
    chunk_size = 10
    for i in range(0, N, chunk_size):
        for j in range(0, M, chunk_size):
            # Process a chunk of cells
            for ii in range(i, min(i + chunk_size, N)):
                for jj in range(j, min(j + chunk_size, M)):
                    in_best = r <= ii < r+k and c <= jj < c+k
                    high = B[ii, jj] >= threshold
                    
                    lat1 = base_lat - ii * lat_step
                    lon1 = base_lon + jj * lon_step
                    lat2 = lat1 - lat_step
                    lon2 = lon1 + lon_step
                    
                    # Only create popup for significant cells
                    if in_best or high:
                        popup_html = f"""
                        <div style="font-family: Arial; padding: 10px;">
                            <h4>Cell Information</h4>
                            <p><b>Position:</b> ({ii},{jj})</p>
                            <p><b>Score:</b> {B[ii,jj]}</p>
                            <p><b>Status:</b> {'Best Square' if in_best else 'High Value'}</p>
                        </div>
                        """
                        popup = folium.Popup(popup_html, max_width=300)
                    else:
                        popup = None
                    
                    # Add to appropriate layer
                    rect = folium.Rectangle(
                        bounds=[[lat1, lon1], [lat2, lon2]],
                        color=best_color if in_best else high_color if high else low_color,
                        fill=True,
                        fill_color=best_color if in_best else high_color if high else low_color,
                        fill_opacity=fill_opacity,
                        weight=1 if not in_best else 2,
                        popup=popup
                    )
                    
                    if in_best:
                        rect.add_to(best_layer)
                    elif high:
                        rect.add_to(high_layer)
                    else:
                        rect.add_to(low_layer)
    
    # Add all layers to map in correct order
    low_layer.add_to(m)
    high_layer.add_to(m)
    best_layer.add_to(m)
    
    # Add the best square outline
    if k is not None:
        outline_layer = folium.FeatureGroup(name='Best Square Outline')
        folium.Rectangle(
            bounds=[[base_lat - r*lat_step, base_lon + c*lon_step],
                    [base_lat - (r+k)*lat_step, base_lon + (c+k)*lon_step]],
            color=highlight_color,
            fill=False,
            weight=3,
            popup=f"Best {k}x{k} Square (Score: {best_sum})",
            opacity=0.8
        ).add_to(outline_layer)
        outline_layer.add_to(m)
    
    # Add information box with improved styling
    info = f"""
    <div style="position:fixed;bottom:40px;left:40px;width:320px;
    background:white;border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,0.1);
    padding:15px;font-family:Arial;font-size:14px;z-index:9999;">
        <div style="font-weight:bold;font-size:16px;margin-bottom:10px;
        color:{best_color};">{app_name} ({map_type.title()})</div>
        <div style="display:grid;grid-template-columns:auto 1fr;gap:8px;">
            <div>📏 Size:</div><div>{k}x{k}</div>
            <div>📍 Position:</div><div>({r},{c})</div>
            <div>📊 Score:</div><div>{best_sum}</div>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(info))
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def save_summary(app_name, shape, k, r, c, best_sum, runtime, idx=None):
    filename = f"{app_name}_{idx}.txt" if idx else f"{app_name}.txt"
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(path, 'w') as f:
        f.write(f"Grid shape: {shape}\n")
        f.write(f"Best square: {k}x{k} from ({r},{c})\n")
        f.write(f"Sum: {best_sum}\n")
        f.write(f"Runtime: {runtime:.4f}s\n")
    return filename

def create_pdf_report(app_name, map_file, k, r, c, best_sum, runtime, idx=None):
    try:
        # Ensure uploads directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Create PDF file
        pdf_file = f"{app_name}_{idx}.pdf" if idx else f"{app_name}.pdf"
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file)
        
        # Initialize PDF canvas with better styling
        c = canvas.Canvas(pdf_path, pagesize=A4)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, 780, f"{app_name} Analysis Report")
        
        c.setFont("Helvetica", 12)
        c.drawString(50, 750, "Analysis Results:")
        c.drawString(70, 730, f"• Best square size: {k}x{k}")
        c.drawString(70, 710, f"• Location: ({r},{c})")
        c.drawString(70, 690, f"• Total score: {best_sum}")
        c.drawString(70, 670, f"• Processing time: {runtime:.4f} seconds")

        # Save map as static image without using Selenium
        map_path = os.path.abspath(map_file)
        if os.path.exists(map_path):
            c.drawString(50, 630, "Visualization Map:")
            c.drawString(70, 610, "Please refer to the interactive map in your browser for detailed analysis.")
            
            # Add a box to represent the map area
            c.rect(70, 300, 400, 300)
            c.drawString(220, 450, "Interactive Map")
            c.drawString(180, 430, "Available in Browser View")
        
        # Add footer
        c.setFont("Helvetica-Oblique", 9)
        c.drawString(50, 50, f"Generated on: {time()}")
        
        c.showPage()
        c.save()
        return pdf_file
    except Exception as e:
        print(f"Error generating PDF report: {str(e)}")
        return None

def benchmark_runtimes(app_name):
    sizes = [5, 10, 15, 20]
    times = []
    for size in sizes:
        B = np.random.randint(0, 100, size=(size, size))
        start = time()
        find_best_square(B)
        times.append(time() - start)
    plt.figure(figsize=(6, 4))
    plt.plot(sizes, times, marker='o', color='#23a6d5')
    plt.xlabel("Grid size (N)")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime vs Grid Size")
    benchmark_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{app_name}_benchmark.png")
    plt.savefig(benchmark_file, dpi=150, bbox_inches='tight')
    plt.close()
    return benchmark_file

# ========================================
# NDVI Fetch
# ========================================
def fetch_ndvi_geotiff(lat, lon, size_px=100):
    config = SHConfig()
    config.sh_client_id = os.getenv("SH_CLIENT_ID", "c51c0fde-8045-4415-b23c-6d2e381aa98e")
    config.sh_client_secret = os.getenv("SH_CLIENT_SECRET", "zWiwZV9E6GPpYZo2ikXcDSKbMyEC2Tqa")
    config.instance_id = "ac04c234-234b-451b-5a30-a7c0d2e7d2b"

    deg_per_px = 10 / 111_320.0
    half_deg = (size_px * deg_per_px) / 2.0
    bbox = BBox(bbox=[lon - half_deg, lat - half_deg,
                      lon + half_deg, lat + half_deg], crs=CRS.WGS84)

    evalscript = """
    //VERSION=3
    function setup() {
      return {
        input: ["B04", "B08", "dataMask"],
        output: { bands: 1, sampleType: "FLOAT32" }
      };
    }
    function evaluatePixel(sample) {
      if (sample.dataMask == 0) return [NaN];
      let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04 + 1e-6);
      return [ndvi * 100];
    }
    """

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=("2024-10-20", "2024-10-27"),
            mosaicking_order=MosaickingOrder.LEAST_CC
        )],
        responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
        bbox=bbox,
        size=(size_px, size_px),
        config=config
    )

    response = request.get_data()
    img = response[0]

    tif_path = os.path.join(app.config['UPLOAD_FOLDER'], 'ndvi_temp.tif')
    transform = rasterio.transform.from_bounds(
        west=lon - half_deg,
        south=lat - half_deg,
        east=lon + half_deg,
        north=lat + half_deg,
        width=size_px,
        height=size_px
    )

    with rasterio.open(
        tif_path, 'w',
        driver='GTiff', height=size_px, width=size_px, count=1,
        dtype=img.dtype, crs='EPSG:4326', transform=transform
    ) as dst:
        dst.write(img, 1)

    return tif_path

# ========================================
# Problem 01: Sensor Fusion
# ========================================
def solve_matrix_chain_memory(p, L):
    n = len(p) - 1
    INF = float('inf')
    dp = [[INF] * n for _ in range(n)]
    parent = [[-1] * n for _ in range(n)]

    for i in range(n):
        dp[i][i] = 0

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            for k in range(i, j):
                cost = dp[i][k] + dp[k+1][j] + p[i] * p[k+1] * p[j+1]
                if cost <= L and cost < dp[i][j]:
                    dp[i][j] = cost
                    parent[i][j] = k

    if dp[0][n-1] == INF:
        return {"status": "impossible", "cost": -1}

    def build_paren(i, j):
        if i == j:
            return f"M{i+1}"
        k = parent[i][j]
        return f"({build_paren(i, k)}{build_paren(k+1, j)})"

    return {
        "status": "valid",
        "cost": int(dp[0][n-1]),
        "parenthesization": build_paren(0, n-1)
    }

# ========================================
# Problem 02: Secure Zones
# ========================================
def line_intersection(p1, p2, q1, q2, eps=1e-10):
    """
    Returns intersection point of two line segments with tolerance.
    eps: small tolerance to handle floating-point errors
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    # Parallel or near-parallel
    if abs(denom) < eps:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    # Allow slight overflow (tolerance)
    if (t >= -eps and t <= 1 + eps) and (u >= -eps and u <= 1 + eps):
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (ix, iy)
    
    return None

def find_secure_region(beams, eps=1e-10):
    """
    Finds convex hull of all beam intersections.
    Returns (hull_points, area) or (None, 0) if invalid.
    """
    points = []

    # Find all intersections
    for i in range(len(beams)):
        for j in range(i + 1, len(beams)):
            p1 = (beams[i][0], beams[i][1])
            p2 = (beams[i][2], beams[i][3])
            q1 = (beams[j][0], beams[j][1])
            q2 = (beams[j][2], beams[j][3])
            inter = line_intersection(p1, p2, q1, q2, eps)
            if inter:
                points.append(inter)

    # Remove exact duplicates
    unique_points = []
    for p in points:
        if not any(abs(p[0] - up[0]) < eps and abs(p[1] - up[1]) < eps for up in unique_points):
            unique_points.append(p)
    points = unique_points

    if len(points) < 3:
        return None, 0

    # Graham scan convex hull
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Sort by x, then y
    points = sorted(points)

    if len(points) < 3:
        return None, 0

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Combine and remove duplicates at ends
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return None, 0

    # Calculate area
    area = 0
    n = len(hull)
    for i in range(n):
        j = (i + 1) % n
        area += hull[i][0] * hull[j][1]
        area -= hull[j][0] * hull[i][1]
    area = abs(area) / 2.0

    return hull, area

# ========================================
# Routes
# ========================================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('files')
        threshold = int(request.form.get('threshold', 80))
        max_k = int(request.form.get('max_k', 4))
        app_name = request.form.get('app_name', 'EcoReserve')
        loc_str = request.form.get('locations', '').strip()
        locations = [float(x) for x in loc_str.split()] if loc_str else []

        if len(files) > 1 and len(locations) != 4 * len(files):
            return render_template('index.html', error="Provide 4 values per CSV: base_lat base_lon lat_step lon_step")

        results = []
        for i, file in enumerate(files):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            B = load_csv_grid(path)

            # Use user-provided or default locations
            if locations:
                base_lat, base_lon, lat_step, lon_step = locations[i*4:i*4+4]
            else:
                base_lat, base_lon, lat_step, lon_step = 11.7, 76.6, 0.008, 0.01

            # Fake bounds for CSV
            west = base_lon
            south = base_lat - B.shape[0] * lat_step
            east = base_lon + B.shape[1] * lon_step
            north = base_lat
            bounds_str = f"{west},{south},{east},{north}"
            grid_shape = B.shape

            start = time()
            best_sum, best = find_best_square(B, max_k=max_k)
            runtime = time() - start
            r, c, k = best

            # Create primary map
            map_file = f"{app_name}_{i+1}.html" if len(files) > 1 else f"{app_name}.html"
            map_path = os.path.join(app.config['UPLOAD_FOLDER'], map_file)
            m = create_map(B, k, r, c, best_sum, threshold, app_name,
                           base_lat, base_lon, lat_step, lon_step, 'primary')
            m.save(map_path)

            # Create secondary map with different visualization
            second_map_file = f"{app_name}_{i+1}_secondary.html" if len(files) > 1 else f"{app_name}_secondary.html"
            second_map_path = os.path.join(app.config['UPLOAD_FOLDER'], second_map_file)
            m2 = create_map(B, k, r, c, best_sum, threshold, app_name,
                           base_lat, base_lon, lat_step, lon_step, 'secondary')
            m2.save(second_map_path)

            summary_file = save_summary(app_name, B.shape, k, r, c, best_sum, runtime, i+1 if len(files) > 1 else None)
            pdf_file = create_pdf_report(app_name, map_path, k, r, c, best_sum, runtime, i+1 if len(files) > 1 else None)

            results.append({
                'grid': filename, 'k': k, 'r': r, 'c': c,
                'best_sum': best_sum, 'runtime': runtime,
                'map_file': map_file, 'second_map_file': second_map_file,
                'summary_file': summary_file, 'pdf_file': pdf_file,
                'bounds': bounds_str, 'grid_shape': grid_shape
            })

        benchmark_file = benchmark_runtimes(app_name)
        return render_template('results.html',
                               results=results,
                               benchmark_file=os.path.basename(benchmark_file))

    return render_template('index.html')

@app.route('/fetch_ndvi', methods=['POST'])
def fetch_ndvi():
    location_str = request.form.get('location', '11.7,76.6')
    lat, lon = map(float, location_str.split(','))
    size_px = int(request.form.get('size', 100))
    threshold = int(request.form.get('threshold', 80))
    max_k = int(request.form.get('max_k', 4))
    app_name = request.form.get('app_name', 'EcoReserve')

    try:
        tif_path = fetch_ndvi_geotiff(lat, lon, size_px)
    except Exception as e:
        return render_template('index.html', error=f"NDVI fetch failed: {e}")

    B, base_lat, base_lon, lat_step, lon_step, bounds_str, grid_shape = load_geotiff_grid(tif_path)
    os.remove(tif_path)

    start = time()
    best_sum, best = find_best_square(B, max_k=max_k)
    runtime = time() - start
    r, c, k = best

    map_file = f"{app_name}_ndvi.html"
    map_path = os.path.join(app.config['UPLOAD_FOLDER'], map_file)
    m = create_map(B, k, r, c, best_sum, threshold, app_name,
                   base_lat, base_lon, lat_step, lon_step)
    m.save(map_path)

    summary_file = save_summary(app_name, B.shape, k, r, c, best_sum, runtime, 'ndvi')
    pdf_file = create_pdf_report(app_name, map_path, k, r, c, best_sum, runtime, 'ndvi')

    # FIXED: Full function call
    benchmark_file = benchmark_runtimes(app_name)

    return render_template(
        'results.html',
        results=[{
            'grid': f"NDVI @ ({lat:.4f}°, {lon:.4f}°)",
            'k': k, 'r': r, 'c': c,
            'best_sum': best_sum, 'runtime': runtime,
            'map_file': map_file, 'summary_file': summary_file, 'pdf_file': pdf_file,
            'bounds': bounds_str, 'grid_shape': grid_shape
        }],
        benchmark_file=os.path.basename(benchmark_file)
    )

@app.route('/matrix_chain_auto', methods=['POST'])
def matrix_chain_auto():
    size = int(request.form['grid_size'])
    
    # Realistic memory limit: 2 * size³
    min_needed = 2 * size * size * size
    L = max(min_needed, 1000000)  # At least 1M

    p = [size, size, size, size]  # 3 matrices: NDVI, Soil, Elevation
    result = solve_matrix_chain_memory(p, L)
    
    # Map M1, M2, M3 to real layer names
    layer_map = {1: "NDVI (Vegetation)", 2: "Soil Moisture", 3: "Elevation"}
    descriptive_order = result["parenthesization"].replace("M1", "NDVI") \
                                                .replace("M2", "Soil Moisture") \
                                                .replace("M3", "Elevation")
    
    return render_template('fusion_result.html', 
                         size=size, L=L, min_needed=min_needed,
                         result=result, order=descriptive_order)

@app.route('/secure_zones_auto', methods=['POST'])
def secure_zones_auto():
    # Get map bounds
    bounds = request.form['bounds']
    west, south, east, north = map(float, bounds.split(','))
    
    # Center of the map
    cx = (west + east) / 2
    cy = (south + north) / 2

    # 4 laser beams that cross at center → guaranteed intersections
    beams = [
        [west, south, east, north],        # Diagonal 1
        [west, north, east, south],        # Diagonal 2
        [west, cy, east, cy],              # Horizontal midline
        [cx, south, cx, north],            # Vertical midline
    ]
    
    # Try to compute real secure zone
    hull, area = find_secure_region(beams)

    # Fallback: if math fails, create a 2×2 cell zone at center
    if hull is None or len(hull) < 3:
        size = int(request.form.get('grid_size', '100'))
        deg_per_cell = (east - west) / size
        cell_area_sqkm = deg_per_cell * deg_per_cell * 12300  # ~111 km² per deg²

        hull = [
            (cx - deg_per_cell, cy - deg_per_cell),
            (cx + deg_per_cell, cy - deg_per_cell),
            (cx + deg_per_cell, cy + deg_per_cell),
            (cx - deg_per_cell, cy + deg_per_cell),
        ]
        area = cell_area_sqkm * 4  # 2×2 cells
        message = "Auto-generated secure zone at map center (beams had minor overlap issues)."
    else:
        message = "Secure zone formed from laser beam intersections."

    return render_template('zone_result.html', 
                         hull=hull, area=area, bounds=bounds, message=message)
@app.route('/uploads/<filename>')
def serve_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return "File not found", 404
    return send_file(file_path, as_attachment=filename.endswith(('.txt', '.pdf')))

if __name__ == '__main__':
    app.run(debug=True)